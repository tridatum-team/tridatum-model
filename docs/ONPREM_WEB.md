# TRID PDF 근거 확인 서비스 설치·운영 안내서 (Django판)

대상 독자: 이 서비스를 설치하고 운영하는 기관 전산 담당자.

이 문서는 Django로 다시 만든 판(`onprem_web`)의 안내서입니다. 화면, 문구, 동작, 제한값, 이용 기록 항목은 기존 판(`docs/ONPREM.md`, 묶음 안 `ONPREM.md`)과 같습니다. 결과를 읽는 법과 한계(1-3절), 운영(4절), 문제 해결(8절)은 `ONPREM.md`를 그대로 따르고, 이 문서에는 설치·명령·파일 이름이 다른 부분만 적습니다.

## 1. 한 서버에는 한 판만 설치

**한 서버에는 한 판만 설치하십시오.** 두 판 모두 443 포트를 쓰므로 함께 띄울 수 없습니다. 기존 판에서 옮겨 오는 경우 기존 판을 `docker compose down`으로 내린 뒤 설치합니다.

- 계정과 이용 기록은 자동 이전하지 않습니다. 기존 판의 `data/onprem.db`와 Django판의 `data/web.db`는 서로 다른 파일입니다. **기존 계정·비밀번호·이용 기록을 유지하려면 새 앱을 시작하거나 관리자를 만들기 전에 9절의 `import-legacy`를 실행하십시오.** 원본 DB는 변경하지 않습니다.
- 이미지와 compose 프로젝트 이름이 다릅니다. 기존 판은 `trid-onprem`, `trid-onprem-app:latest`이고 Django판은 `trid-onprem-web`, `trid-onprem-web-app:latest`입니다. 모델 서버 이미지 `trid-onprem-llm:latest`와 Caddy 이미지는 같습니다.
- 설정 파일 `onprem.json`의 키는 기존 판과 같습니다. 기존 판 설정을 그대로 복사해 써도 됩니다.

## 2. 권장 사양

기존 판과 같습니다(`ONPREM.md` 2절). x86_64 리눅스, AVX2 이상 지원 CPU, Docker Engine과 Docker Compose v2가 필요합니다. 동시 5~10명 기준 권장 수치는 빌드 머신 리허설 후 `release_check.py` 측정값으로 기입합니다. 추정값을 쓰지 않습니다. `app` 컨테이너는 `mem_limit: 1g`로 고정입니다.

## 3. 설치

### 3-1. 배포 묶음 확인

배포 담당자가 전달하는 폴더 `trid-onprem-web-<커밋>/`에는 이미지 tar(`trid-onprem-web-<커밋>.tar`, 이미지 3개), `IMAGES.txt`, `SHA256SUMS`, `compose.yaml`, `Caddyfile`, `onprem.example.json`, `make_self_signed.sh`, `release_check.py`와 점검 자료, `LICENSES.md`, 이 안내서(`ONPREM_WEB.md`), 기존 판 안내서(`ONPREM.md`), 빈 `certs/`, `data/` 폴더가 들어 있습니다.

```sh
cd trid-onprem-web-<커밋>
sha256sum -c SHA256SUMS      # 모든 줄이 OK여야 합니다
docker load -i trid-onprem-web-<커밋>.tar
docker images | grep -E 'trid-onprem|caddy'
```

`trid-onprem-web-app:latest`, `trid-onprem-llm:latest`, `caddy:2.8.4` 세 개가 보여야 합니다.

### 3-2. 호스트 이름, 설정 파일, `data` 폴더

```sh
echo 'TRID_HOSTNAME=trid.example.internal' > .env
cp onprem.example.json data/onprem.json      # hostname을 .env의 TRID_HOSTNAME과 같게 고칩니다
sudo chown -R 10001:10001 data              # 앱은 uid 10001로 실행됩니다
sudo chmod 700 data                         # data/web.db에는 비밀번호 해시와 접속 IP가 들어 있습니다
```

`hostname`, `.env`의 `TRID_HOSTNAME`, 인증서 이름(SAN), 브라우저 접속 주소는 모두 같아야 합니다. 다르면 로그인을 포함한 모든 POST가 "허용되지 않은 요청입니다."로 실패합니다. 업로드 제한과 `TRID_MAX_BODY`의 관계, 보관 기간(`retention_days`) 등 나머지 설정은 `ONPREM.md` 3-3, 3-4절과 같습니다.

### 3-3. 인증서

`certs/cert.pem`(인증서, 필요하면 중간 인증서 포함)과 `certs/key.pem`(개인 키)을 넣습니다. 키는 root 소유 모드 600이어야 합니다.

```sh
chmod 600 certs/key.pem
chown 0:0 certs/key.pem
```

시험 설치라면 `sudo ./make_self_signed.sh trid.example.internal certs`로 자체 서명 인증서(유효기간 825일)를 만들 수 있습니다.

### 3-4. 시작

```sh
docker compose up -d
docker compose ps             # llm, app, tls 모두 running이어야 합니다
docker compose logs app       # "서버를 시작합니다"가 보이면 정상
```

앱은 시작할 때 데이터베이스 마이그레이션을 자동으로 적용합니다. 처음 시작하면 빈 `data/`에 `web.db`가 만들어집니다. 로그에는 질문, 파일 이름, 답변, PDF 내용이 남지 않습니다(예외 종류 이름이나 상태 코드만 기록).

### 3-5. 첫 관리자 만들기

기존 계정 이전을 완료했다면 이 절은 건너뛰고 기존 관리자 계정으로 로그인합니다. 새 설치에만 아래 명령을 사용합니다.

```sh
docker compose exec app python -I -m onprem_web create-admin --config /data/onprem.json --username <계정이름>
```

비밀번호를 두 번 묻습니다(최소 10자). 스크립트에서 쓸 때는 `--password-stdin`을 붙이면 표준 입력 한 줄을 비밀번호로 씁니다(명령줄에 비밀번호를 쓰지 마십시오).

```sh
printf '%s\n' "$ADMIN_PASSWORD" | docker compose exec -T app python -I -m onprem_web create-admin \
    --config /data/onprem.json --username <계정이름> --password-stdin
```

이렇게 만든 첫 관리자는 비밀번호 변경을 강제하지 않습니다. 이후 계정은 `https://<호스트 이름>/admin` 관리 화면에서 만들며, 첫 로그인 때 비밀번호를 바꿔야 합니다.

### 3-6. 배포 전 점검

기존 판과 같은 `release_check.py`를 고치지 않고 그대로 씁니다. 방법과 판단 기준은 `ONPREM.md` 3-8절과 같습니다.

```sh
python3 release_check.py --base https://trid.example.internal --user <관리자 계정> \
    --pdf release-document.pdf --questions release-questions.json --out report.json
```

## 4. 운영

계정 관리, 이용 기록 조회와 CSV, 세션 규칙은 `ONPREM.md` 4절과 같습니다. 다른 점은 아래뿐입니다.

- CSV 내려받기는 관리 화면의 "CSV 내려받기" 버튼으로만 합니다(주소는 같은 `/api/admin/events.csv`이고, 보안 토큰을 주소에 넣지 않도록 POST로 요청합니다). 파일 이름은 기존 판과 같은 `trid-usage-YYYYMMDD.csv`입니다.
- 이용 기록의 시각은 브라우저가 있는 PC의 시간대로 표시됩니다(컨테이너 시간대와 무관).
- 수동 정리: `docker compose exec app python -I -m onprem_web purge --config /data/onprem.json`. 보관 기간이 지난 기록은 시작할 때와 이후 24시간마다 자동으로 지워지고 `retention_purge`로 기록됩니다.
- 유휴 로그인(30분)과 그 메모리 문서는 1분마다 정리됩니다.

## 5. 백업·복구

백업 대상은 `data/` 폴더입니다. 계정, 세션, 이용 기록은 `data/web.db`(SQLite, 같은 폴더의 `web.db-wal`, `web.db-shm` 포함)에, 설정은 `data/onprem.json`에, 세션 서명 키는 `data/web-secret-key`에 있습니다. 인증서(`certs/`)와 `.env`도 함께 보관하십시오.

**방법 A. 서비스를 멈추고 복사(권장).**

```sh
docker compose stop app
sudo cp -a data data-backup-$(date +%Y%m%d)
docker compose start app
```

**방법 B. SQLite 백업.**

```sh
docker compose stop app
sudo sqlite3 data/web.db ".backup 'backup-web.db'"
sudo cp data/onprem.json backup-onprem.json
docker compose start app
```

**복구.**

```sh
docker compose stop app
sudo cp -a data-backup-<날짜>/. data/        # 방법 B 파일이면 backup-web.db를 data/web.db로 복사
sudo rm -f data/web.db-wal data/web.db-shm     # 방법 B 파일을 복원할 때만
sudo chown -R 10001:10001 data
sudo chmod 700 data
docker compose start app
```

백업 파일에는 비밀번호 해시와 이용 기록(IP 포함)이 들어 있으므로 접근을 제한하십시오. `web-secret-key`가 바뀌거나 없어지면 로그인한 사람들의 세션만 끝나고(다시 로그인), 계정과 기록은 그대로입니다.

## 6. 업그레이드·되돌리기

1. 5절 방법 A로 `data/`를 백업합니다.
2. 새 묶음을 반입하고 `sha256sum -c SHA256SUMS`로 확인합니다.
3. 운영 중인 폴더에서 `docker compose down`.
4. 새 폴더에 기존 `data/`, `certs/`, `.env`를 옮깁니다.
5. 새 폴더에서 `docker load -i trid-onprem-web-<새 커밋>.tar` 후 `docker compose up -d`. 새 판에 데이터베이스 변경이 있으면 시작할 때 마이그레이션이 자동으로 적용됩니다. 따로 실행할 명령은 없습니다.
6. 로그인과 3-6절 점검을 다시 실행합니다.

**되돌리기.** 마이그레이션은 앞으로만 적용되므로, 이전 판으로 되돌릴 때는 **반드시 업그레이드 전 백업으로 `data/`를 복구**한 뒤 이전 폴더에서 `docker load -i trid-onprem-web-<이전 커밋>.tar`, `docker compose up -d` 합니다. 이전 폴더와 tar는 지우지 말고 보관하십시오.

## 7. 인증서 교체

```sh
docker compose stop tls
cp <새 인증서> certs/cert.pem
cp <새 키> certs/key.pem
chmod 600 certs/key.pem
chown 0:0 certs/key.pem
docker compose up -d tls
```

호스트 이름이 바뀌면 `.env`의 `TRID_HOSTNAME`, `data/onprem.json`의 `hostname`, 인증서 SAN을 모두 새 이름으로 맞추고 `docker compose up -d`, `docker compose restart app`을 실행하십시오.

## 8. 개발자용: 테스트 실행

저장소 루트에서 실행합니다. 개발 환경은 `uv venv .venv --python 3.12`, `uv pip install -r deploy/onprem_web/requirements-dev.txt`, `uv pip install --no-deps -e .`로 만듭니다(`uv run`이나 `uv sync`는 잠금 파일에 맞춰 Django를 지우므로 쓰지 않습니다).

```sh
.venv/Scripts/python -m unittest discover -s tests -p 'test_webapp_*.py'   # Django판 (Linux: .venv/bin/python)
.venv/Scripts/python -m unittest discover -s tests -p 'test_onprem_*.py'   # 기존 판
```

`test_webapp_parity.py`는 두 판에 같은 요청을 보내 응답이 같은지 확인하고, `release_check.py`를 고치지 않고 Django판에 대해 실행합니다. 이미지 빌드와 실제 설치본 점검은 Docker가 있는 빌드 머신에서 `deploy/onprem_web/build_bundle.sh`로 묶음을 만든 뒤 3-6절대로 합니다.

## 9. 기존 계정과 이용 기록 이전

원본 앱의 쓰기를 중단하고 SQLite 백업을 만든 뒤 진행합니다. 기존 `onprem.db`를 `web.db`로 이름만 바꾸거나 덮어쓰지 마십시오. 설정 파일의 `data_dir`는 새 Django DB를 만들 디렉터리이며, 원본 DB는 `--source`로 별도 지정합니다.

```sh
umask 077
python -I -m onprem_web import-legacy \
    --config /path/to/onprem.json \
    --source /path/to/onprem.db
```

- 네이티브 스키마 버전 1의 계정·이용 기록을 읽기 전용 스냅샷으로 가져옵니다. 원본 파일은 수정하지 않습니다.
- 계정 이름, 기존 비밀번호 해시, 역할, 비활성화·잠금·비밀번호 변경 요구 상태, 생성 시각을 보존합니다. 기존 비밀번호로 로그인하며, 정상 로그인 시 비밀번호 해시를 Django 형식으로 갱신합니다.
- 이용 기록의 ID·시각·내용과 다음 ID의 기준값을 보존합니다. 원본의 계정·기록을 출력하지 않고 성공 시 `Imported accounts=N audit_events=N`만 표시합니다.
- 대상에 계정·이용 기록 등이 있거나 원본 형식이 지원 범위를 벗어나면 거부합니다. 부분 이전과 기존 대상 덮어쓰기를 허용하지 않습니다. 실패한 경우 원인을 확인하며, 기존 DB를 임의로 삭제하지 마십시오.
- 로그인 세션과 메모리의 PDF는 이전하지 않습니다. 사용자는 다시 로그인하고 PDF를 다시 올려야 합니다.
- 기존 `retention_days` 정책은 바꾸지 않습니다. 가져온 모든 기록을 보존한 뒤에도 서버 시작 시와 매일 실행되는 기존 보관 기간 정리 정책은 적용됩니다.

## 10. 맥미니 테스트 도메인용 네이티브 실행

이 절은 `test.tridatum.co` 전용입니다. 기관 납품용 Linux Compose 배포와 구분하며, 운영 회사 홈페이지나 다른 서비스를 변경하지 않습니다.

### 준비된 파일

- `deploy/onprem_web/requirements-macos-arm64.txt`: Python 3.12 ARM64용 해시 고정 의존성. `pip install --no-deps --require-hashes -r ...`로 설치합니다.
- 앱 wheel을 릴리스 전용 venv에 `--no-deps`로 설치합니다. `python -I`는 현재 폴더의 소스를 가져오지 않으므로 wheel 설치가 필요합니다.
- `deploy/onprem_web/co.tridatum.trid.onprem.plist`: `releases/django-5f45934/venv/bin/python`으로 실행하는 앱 작업. 기존 `data/onprem.json`과 로그 경로를 유지합니다.
- `deploy/onprem_web/Caddyfile.test`: 인증된 터널 요청만 받아 앱의 공개 Host·클라이언트 IP를 지정하는 루프백 게이트웨이.
- `deploy/onprem_web/worker.mjs`, `wrangler.test.json`: Django 화면·정적 파일·폼·CSV POST 경로를 허용하는 테스트 도메인 전용 Worker. 원점·공개 호스트·비밀키 경계를 유지합니다.

프로덕션 보안 설정을 유지한 루프백 실행 명령은 다음과 같습니다.

```sh
python -I -m onprem_web serve --config /path/to/onprem.json --bind 127.0.0.1
```

**공개 배포에 `--insecure-dev`를 사용하지 마십시오.** 기존 모델 프로세스를 소유하는 `co.tridatum.trid.site`도 중지하지 않습니다.

### 적용 순서와 복구 경계

1. SSH 연결 복구 후 실제 앱·게이트웨이의 실행 파일, 설정 경로, 계정 DB와 배포 버전을 확인합니다. 문서만으로 현재 실행 경로를 추정해 덮어쓰지 않습니다.
2. 새 릴리스 venv와 wheel을 별도 경로에 설치하고, 현재 앱·게이트웨이 plist/설정과 Worker 버전을 보관합니다. 모델·터널·기존 릴리스는 유지합니다.
3. 기존 앱만 중지하고 원본 DB를 백업한 뒤 9절의 이전을 실행합니다. 실패하면 새 앱을 시작하지 않고 기존 앱을 복구합니다.
4. 새 앱을 루프백에서 시작하고 실제 이전된 관리자·일반 계정, 잠금 상태, 이용 기록을 확인합니다. Caddy와 Worker는 새 앱의 경로·헤더 계약에 맞춰 함께 전환해야 합니다.
5. 공개 HTTPS에서 로그인·PDF 업로드·질문·관리자 화면·CSV POST를 확인합니다. 직접 원점 접근 차단도 확인한 후 배포 완료로 기록합니다.
6. 공개 전환 전 실패 시 이전 앱·게이트웨이·Worker를 함께 복원합니다. **Django 전환 후 생성·수정된 계정과 기록은 원본 DB에 역동기화되지 않습니다.** 공개 사용 이후 복구가 필요하면 새 `web.db`도 보관하고 데이터 차이를 별도로 처리해야 합니다. 단순 DB 덮어쓰기로 최근 기록을 잃지 마십시오.

### 2026-11-17 검증 상태 — 아직 공개 미배포

- Django 테스트 142개, 기존 네이티브 앱 테스트 127개, Worker 경로 테스트 10개 통과.
- 해시 고정 ARM64 의존성과 앱 wheel을 설치한 별도 venv에서 합성 기존 계정 3개·이용 기록 2개 이전 성공.
- 실제 Chrome에서 이전한 관리자 비밀번호 로그인, PDF 업로드·텍스트 추출, 로컬 Qwen3-1.7B 원본 응답, 관리자 계정·이용 기록 화면을 확인했습니다. CSV POST는 인증된 브라우저 HTTPS 요청으로 HTTP200·CSV 내용 보존을 확인했습니다.
- 실제 Caddy와 앱을 사용했습니다. 앱은 `127.0.0.1:51874`에만 바인딩되며 인증 없는 게이트웨이 요청은 HTTP403이었습니다. 브라우저의 테스트 도메인은 격리 프로필 안에서만 로컬 TLS 포트로 연결했습니다.
- 모델 응답은 `unverified`로 표시되었습니다. 연결·표시 검증이지 답변 품질 합격이 아닙니다.
- 실제 맥미니의 공인 IP·Tailscale IP SSH 연결은 모두 시간초과였습니다. 위 검증은 로컬 리허설이며 공개 Cloudflare/터널 전환과 실제 운영 DB 이전은 수행하지 않았습니다.

