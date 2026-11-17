# TRID PDF 근거 확인 서비스 설치·운영 안내서 (Django판)

대상 독자: 이 서비스를 설치하고 운영하는 기관 전산 담당자.

이 문서는 Django로 다시 만든 판(`onprem_web`)의 안내서입니다. 화면, 문구, 동작, 제한값, 이용 기록 항목은 기존 판(`docs/ONPREM.md`, 묶음 안 `ONPREM.md`)과 같습니다. 결과를 읽는 법과 한계(1-3절), 운영(4절), 문제 해결(8절)은 `ONPREM.md`를 그대로 따르고, 이 문서에는 설치·명령·파일 이름이 다른 부분만 적습니다.

## 1. 한 서버에는 한 판만 설치

**한 서버에는 한 판만 설치하십시오.** 두 판 모두 443 포트를 쓰므로 함께 띄울 수 없습니다. 기존 판에서 옮겨 오는 경우 기존 판을 `docker compose down`으로 내린 뒤 설치합니다.

- 계정과 이용 기록은 옮겨지지 않습니다. 기존 판의 `data/onprem.db`와 Django판의 `data/web.db`는 서로 다른 파일입니다. 기존 판 이용 기록은 기존 판 관리 화면에서 CSV로 내려받아 보관한 뒤 옮기십시오.
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
