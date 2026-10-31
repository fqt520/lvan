#!/usr/bin/env bash
# =====================================================================
#  Lvan 无痕去水印 —— VPS 一键部署脚本
#
#  用法（root 用户执行）：
#    只有 IP：   bash install.sh
#    有域名：    DOMAIN=wm.example.com bash install.sh     # 自动申请 HTTPS 证书
#  再次运行即为升级（保留已下载的模型）。
#
#  做的事情：安装 nginx → 下载网站代码 → 下载 AI 模型和推理引擎到本机
#  （国内访问不再依赖 huggingface / CDN）→ 配置跨源隔离响应头（多线程推理）
#  → 有域名时用 Let's Encrypt 自动配置 HTTPS。
# =====================================================================
set -euo pipefail

REPO="fqt520/lvan"
BRANCH="${BRANCH:-main}"
APP_DIR="${APP_DIR:-/var/www/lvan}"
DOMAIN="${DOMAIN:-}"
EMAIL="${EMAIL:-}"
ORT_VERSION="1.20.1"

say()  { printf '\033[1;35m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[x]\033[0m %s\n' "$*"; exit 1; }

[ "$(id -u)" = 0 ] || die "请用 root 用户运行（或在命令前加 sudo）"

# ---------- 1. 安装 nginx ----------
if command -v apt-get >/dev/null 2>&1; then
  PM=apt
  export DEBIAN_FRONTEND=noninteractive
  say "安装 nginx（apt）…"
  apt-get update -y >/dev/null
  apt-get install -y nginx curl tar ca-certificates >/dev/null
  [ -n "$DOMAIN" ] && apt-get install -y certbot python3-certbot-nginx >/dev/null
elif command -v dnf >/dev/null 2>&1 || command -v yum >/dev/null 2>&1; then
  PM=$(command -v dnf >/dev/null 2>&1 && echo dnf || echo yum)
  say "安装 nginx（$PM）…"
  $PM install -y epel-release >/dev/null 2>&1 || true
  $PM install -y nginx curl tar ca-certificates >/dev/null
  [ -n "$DOMAIN" ] && $PM install -y certbot python3-certbot-nginx >/dev/null
else
  die "暂不支持这个系统，请使用 Ubuntu / Debian / CentOS / Rocky / AlmaLinux"
fi

# ---------- 工具：多个地址依次尝试下载 ----------
fetch() { # fetch <输出文件> <最小字节数> <url1> [url2 ...]
  local out=$1 min=$2; shift 2
  local tmp="$out.part"
  for url in "$@"; do
    say "  下载 $url"
    if curl -fL --retry 2 --connect-timeout 15 --progress-bar -o "$tmp" "$url"; then
      local size; size=$(stat -c %s "$tmp" 2>/dev/null || echo 0)
      if [ "$size" -ge "$min" ]; then mv -f "$tmp" "$out"; return 0; fi
      warn "  文件不完整（$size 字节），换下一个地址"
    fi
  done
  rm -f "$tmp"
  return 1
}

# ---------- 2. 下载网站代码 ----------
say "下载网站代码…"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
fetch "$TMP/src.tar.gz" 10000 \
  "https://codeload.github.com/$REPO/tar.gz/refs/heads/$BRANCH" \
  "https://github.com/$REPO/archive/refs/heads/$BRANCH.tar.gz" \
  "https://ghfast.top/https://github.com/$REPO/archive/refs/heads/$BRANCH.tar.gz" \
  || die "代码下载失败：VPS 无法访问 GitHub"
tar -xzf "$TMP/src.tar.gz" -C "$TMP"
SRC=$(find "$TMP" -maxdepth 1 -mindepth 1 -type d | head -n1)
mkdir -p "$APP_DIR/models" "$APP_DIR/vendor/ort"
rm -rf "$APP_DIR/js" "$APP_DIR/css"
cp -r "$SRC/index.html" "$SRC/sw.js" "$SRC/js" "$SRC/css" "$APP_DIR/"

# ---------- 3. 下载 AI 模型（已存在则跳过） ----------
model() { # model <文件名> <大小> <HF 路径>
  local f="$APP_DIR/models/$1" size=$2 path=$3
  if [ -f "$f" ] && [ "$(stat -L -c %s "$f")" -ge "$size" ]; then say "模型 $1 已存在，跳过"; return; fi
  say "下载模型 $1（$((size / 1048576))MB）…"
  fetch "$f" "$size" "https://hf-mirror.com/$path" "https://huggingface.co/$path" || die "模型 $1 下载失败"
}
model ch_PP-OCRv4_det_infer.onnx 4745517 "SWHL/RapidOCR/resolve/main/PP-OCRv4/ch_PP-OCRv4_det_infer.onnx"
model lama_fp32.onnx 208044816 "Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"

# ---------- 4. 下载推理引擎 ----------
say "下载推理引擎 onnxruntime-web $ORT_VERSION…"
for f in ort.webgpu.min.js ort-wasm-simd-threaded.jsep.mjs ort-wasm-simd-threaded.jsep.wasm; do
  [ -s "$APP_DIR/vendor/ort/$f" ] && continue
  fetch "$APP_DIR/vendor/ort/$f" 1000 \
    "https://registry.npmmirror.com/onnxruntime-web/$ORT_VERSION/files/dist/$f" \
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@$ORT_VERSION/dist/$f" \
    "https://unpkg.com/onnxruntime-web@$ORT_VERSION/dist/$f" || die "推理引擎下载失败"
done
chmod -R a+rX "$APP_DIR"

# ---------- 5. 配置 nginx ----------
say "配置 nginx…"
if [ -n "$DOMAIN" ]; then
  LISTEN="listen 80;"
  NAME="$DOMAIN"
else
  # 只有 IP：作为默认站点，并停用系统自带的默认站点
  LISTEN="listen 80 default_server;"
  [ -s /proc/net/if_inet6 ] && LISTEN="$LISTEN listen [::]:80 default_server;"
  NAME="_"
  [ -e /etc/nginx/sites-enabled/default ] && rm -f /etc/nginx/sites-enabled/default
  if [ -f /etc/nginx/nginx.conf ] && grep -q "default_server" /etc/nginx/nginx.conf; then
    cp -n /etc/nginx/nginx.conf /etc/nginx/nginx.conf.lvan-bak
    sed -i 's/ default_server//g' /etc/nginx/nginx.conf
  fi
fi
CONF=/etc/nginx/conf.d/lvan.conf
cat > "$CONF" <<NGINX
server {
    $LISTEN
    server_name $NAME;
    root $APP_DIR;
    index index.html;

    # 跨源隔离：开启后浏览器才允许多线程推理（快数倍）
    add_header Cross-Origin-Opener-Policy "same-origin" always;
    add_header Cross-Origin-Embedder-Policy "credentialless" always;
    add_header Cross-Origin-Resource-Policy "cross-origin" always;

    gzip on;
    gzip_types text/css application/javascript text/javascript application/wasm;
    sendfile on;

    # 有的系统 nginx 不认识 .mjs，浏览器会拒绝加载模块，这里显式指定
    location ~ \.mjs\$ {
        types { application/javascript mjs; }
        add_header Cross-Origin-Opener-Policy "same-origin" always;
        add_header Cross-Origin-Embedder-Policy "credentialless" always;
        add_header Cross-Origin-Resource-Policy "cross-origin" always;
    }
    location ~ \.(wasm)\$ {
        types { application/wasm wasm; }
        add_header Cross-Origin-Opener-Policy "same-origin" always;
        add_header Cross-Origin-Embedder-Policy "credentialless" always;
        add_header Cross-Origin-Resource-Policy "cross-origin" always;
        add_header Cache-Control "public, max-age=2592000";
    }
    location /models/ {
        types { application/octet-stream onnx; }
        add_header Cross-Origin-Resource-Policy "cross-origin" always;
        add_header Cache-Control "public, max-age=2592000";
    }
    location / {
        try_files \$uri \$uri/ =404;
    }
}
NGINX
nginx -t || die "nginx 配置检查失败"
systemctl enable nginx >/dev/null 2>&1 || true
systemctl restart nginx 2>/dev/null || nginx -s reload 2>/dev/null || nginx || die "nginx 启动失败"

# ---------- 6. 防火墙放行 ----------
if command -v ufw >/dev/null 2>&1 && ufw status | grep -q "Status: active"; then
  ufw allow 80/tcp >/dev/null; ufw allow 443/tcp >/dev/null
fi
if command -v firewall-cmd >/dev/null 2>&1 && firewall-cmd --state >/dev/null 2>&1; then
  firewall-cmd --permanent --add-service=http >/dev/null; firewall-cmd --permanent --add-service=https >/dev/null; firewall-cmd --reload >/dev/null
fi
# SELinux：允许 nginx 读取网站目录
command -v chcon >/dev/null 2>&1 && chcon -R -t httpd_sys_content_t "$APP_DIR" 2>/dev/null || true

# ---------- 7. HTTPS ----------
URL=""
if [ -n "$DOMAIN" ]; then
  say "申请 HTTPS 证书（$DOMAIN）…"
  if [ -n "$EMAIL" ]; then EM=(-m "$EMAIL"); else EM=(--register-unsafely-without-email); fi
  if certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos "${EM[@]}" --redirect; then
    URL="https://$DOMAIN/"
  else
    warn "证书申请失败：请确认域名已解析到本机 IP，且 80 端口对外开放。暂时可用 http://$DOMAIN/ 访问"
    URL="http://$DOMAIN/"
  fi
else
  IP=$(curl -fs4 --connect-timeout 5 https://api.ipify.org 2>/dev/null || curl -fs4 --connect-timeout 5 https://ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}')
  URL="http://$IP/"
fi

echo
say "部署完成！手机/电脑浏览器打开： $URL"
if [ -z "$DOMAIN" ]; then
  warn "当前是 http + IP 访问：浏览器只在 https 下才开放显卡加速和多线程，手机上会比较慢。"
  warn "建议绑定一个域名后运行：DOMAIN=你的域名 bash install.sh"
fi
warn "如果打不开，请在云服务商控制台的「安全组/防火墙」里放行 80 和 443 端口。"
