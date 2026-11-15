#!/usr/bin/env bash
# VM 자동 갱신: Artifact Registry 의 :main 태그가 바뀌었으면 받아서 해당 서비스만 다시 띄운다.
# GitHub Actions 는 이미지만 올리고, VM 이 1분마다 이 스크립트로 가져온다 (VM 접속 키를 GitHub 에 두지 않는다).
#
# /opt/skydetect/autodeploy.env 에 둔다:
#   COMPOSE_FILE=compose.app.yml
#   SERVICES="backend clip-bridge relay-heartbeat"   # 레지스트리 이미지를 쓰는 서비스만 (Docker Hub 이미지는 매번 받지 않는다)
set -euo pipefail
cd /opt/skydetect
# shellcheck disable=SC1091
. ./autodeploy.env

REGISTRY=asia-northeast3-docker.pkg.dev
# VM 서비스 계정 토큰으로 레지스트리에 로그인한다 (토큰은 1시간짜리라 매번 새로 받는다).
curl -sf -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token \
  | python3 -c 'import sys, json; print(json.load(sys.stdin)["access_token"])' \
  | docker login -u oauth2accesstoken --password-stdin "https://$REGISTRY" >/dev/null

env_args=()
[ -f .env ] && env_args=(--env-file .env)

before=$(docker compose -f "$COMPOSE_FILE" "${env_args[@]}" images -q $SERVICES 2>/dev/null | sort | tr '\n' ' ')
# shellcheck disable=SC2086
docker compose -f "$COMPOSE_FILE" "${env_args[@]}" pull --quiet $SERVICES
# up -d 는 이미지가 바뀐 컨테이너만 다시 만든다.
# shellcheck disable=SC2086
docker compose -f "$COMPOSE_FILE" "${env_args[@]}" up -d --no-deps $SERVICES >/dev/null 2>&1
after=$(docker compose -f "$COMPOSE_FILE" "${env_args[@]}" images -q $SERVICES 2>/dev/null | sort | tr '\n' ' ')
if [ "$before" != "$after" ]; then
  echo "updated: $SERVICES"
  docker image prune -f >/dev/null
fi
