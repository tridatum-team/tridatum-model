# TRID PDF 근거 확인 서비스 설치·운영 안내서 (온프레미스판)

대상 독자: 이 서비스를 설치하고 운영하는 기관 전산 담당자.
서비스 한 줄 요약: 직원이 웹에서 텍스트 PDF 한 건을 올리고 질문하면, 그 PDF 안에서 찾은 원문 인용을 쪽 번호와 함께 보여 주는 내부망 전용 서비스입니다. 외부 인터넷에 연결하지 않고 동작합니다.

## 1. 구성과 한계

### 1-1. 구성

컨테이너 3개가 `docker compose`로 함께 동작합니다.

| 서비스 | 역할 | 외부 노출 |
| --- | --- | --- |
| `tls` (Caddy 2.8.4) | HTTPS 종단, 요청 본문 제한(기본 3MB, `.env`의 `TRID_MAX_BODY`), 클라이언트 IP를 `X-Real-IP`로 앱에 전달 | 443 포트만 공개 |
| `app` | 웹 화면, 로그인, PDF 읽기, 이용 기록 (SQLite) | 내부 네트워크만 |
| `llm` | llama.cpp 서버와 모델 파일(Qwen3-1.7B Q8_0) | 내부 네트워크만 |

- 호스트에서 열리는 포트는 443 하나입니다. `app`과 `llm`은 내부 네트워크(`internal: true`)에만 있어 서버 밖에서 직접 접근할 수 없습니다.
- 클라이언트가 보낸 `X-Forwarded-*` 헤더는 Caddy가 지우고, 접속 IP는 Caddy가 `X-Real-IP`로 다시 채웁니다. 이용 기록의 IP는 이 값입니다.
- 설치와 실행 중 외부 네트워크로 나가는 통신은 없습니다(모델 내려받기, 원격 측정, 웹 폰트, CDN 모두 없음).

### 1-2. 모델 상태 문구

화면 상단과 로그인 화면에 아래 문구가 그대로 표시됩니다. 임의로 바꾸거나 가리지 마십시오.

> 실험 서비스 · 원본 모델(미특화) · 품질 기준 미달

모델은 Qwen3-1.7B 원본 가중치(Apache-2.0)이며 업무용으로 추가 학습하지 않았습니다. 화면의 "모델 정보" 카드에 근거 모델, 라이선스, 측정 결과와 한계가 적혀 있습니다.

### 1-3. 결과를 읽는 법과 한계

- "확인된 인용"은 올린 PDF에서 글자 그대로 일치하는 문장입니다. 원문에 그 표현이 있다는 뜻일 뿐, 질문의 정답이라는 뜻이 아닙니다.
- "검증되지 않은 해설"은 모델이 쓴 문장 중 원문 일치가 확인되지 않은 부분입니다. 반드시 원문(쪽 번호)과 대조해야 하며, 공문서·결재 근거로 그대로 쓰지 마십시오.
- 관련도 상위 구간(최대 6개)만 확인합니다. "찾지 못했다"는 결과가 문서의 다른 곳에 정보가 없다는 보장은 아닙니다. 근거를 찾지 못한 질문에 모델이 기억으로 답하지 않습니다.
- 질문마다 독립 처리합니다. 앞선 질문과 답을 기억하지 못하므로 "그럼 그건?" 같은 이어 묻기는 되지 않습니다.
- 이미지·스캔 PDF는 지원하지 않습니다(텍스트가 들어 있는 PDF만). 표는 행·열 대응을 검증하지 않습니다.
- 한 번에 한 건의 생성만 처리합니다. 동시에 질문이 겹치면 "다른 질문을 처리 중입니다. 잠시 후 다시 시도해 주세요."가 표시되고 다시 시도하면 됩니다. 동시 사용자는 5~10명 정도를 기준으로 합니다(로그인 세션은 최대 32개, 세션당 분당 요청 20회).
- 업로드 제한(기본값, `data/onprem.json`의 설정으로 변경 가능): 파일 2MiB, 20쪽, 추출 글자 50,000자, 읽기 15초. 초과하면 화면에 사유가 표시됩니다.

## 2. 권장 사양

필수 조건: x86_64 리눅스, CPU가 AVX2 이상 지원(이미지는 `GGML_NATIVE=OFF`, AVX2 고정으로 빌드됨), Docker Engine과 Docker Compose v2.

동시 5~10명 기준 권장 수치는 아래 표에 빌드 머신 리허설 후 기입합니다. 추정값을 쓰지 않습니다.

| 항목 | 권장 | 비고 |
| --- | --- | --- |
| CPU 코어 | 실측 필요 (빌드 머신 리허설 후 기입) | `.env`의 `LLM_THREADS`(기본 4)로 모델 서버 스레드 수 조정 |
| 메모리 | 실측 필요 (빌드 머신 리허설 후 기입) | `app` 컨테이너는 `mem_limit: 1g` 고정 |
| 디스크 | 실측 필요 (빌드 머신 리허설 후 기입) | 이미지 tar 크기와 `data/` 증가량 합산 |
| 질문당 처리 시간 | 실측 필요 (빌드 머신 리허설 후 기입) | |

실측 방법: 설치 후 3-8절의 `release_check.py`를 실행하면 `report.json`에 질문별 `seconds`와 요약(`median_seconds`, `max_seconds`)이 기록됩니다. 대상 사양과 같은 서버에서 3회 이상 돌려 중앙값과 최대값을 이 표에 적습니다. 메모리와 CPU는 점검을 돌리는 동안 `docker stats`로 관찰합니다. 디스크는 `docker load` 전후의 `docker system df`와 `du -sh data`로 확인합니다.

## 3. 설치

### 3-1. 배포 묶음 확인

배포 담당자가 전달하는 폴더 `trid-onprem-<커밋>/`의 내용입니다.

| 파일 | 설명 |
| --- | --- |
| `trid-onprem-<커밋>.tar` | 이미지 3개(앱, 모델 서버, Caddy) |
| `IMAGES.txt` | 이미지 ID·다이제스트, 기반 이미지, llama.cpp 커밋, 모델·토크나이저 sha256 |
| `SHA256SUMS` | 묶음 안 모든 파일의 sha256 |
| `compose.yaml`, `Caddyfile` | 실행 구성 (수정하지 않음) |
| `onprem.example.json` | 설정 파일 견본 |
| `make_self_signed.sh` | 자체 서명 인증서 생성 스크립트 |
| `release_check.py`, `release-questions.json`, `release-document.txt` | 배포 전 점검 도구와 가상 점검 자료 (3-8절) |
| `LICENSES.md` | 제3자 라이선스 |
| `ONPREM.md` | 이 안내서 |
| `certs/`, `data/` | 빈 폴더 |

반입 후 무결성을 먼저 확인합니다.

```sh
cd trid-onprem-<커밋>
sha256sum -c SHA256SUMS      # 모든 줄이 OK여야 합니다
cat IMAGES.txt               # 배포 담당자가 전달한 값과 대조(선택)
```

하나라도 OK가 아니면 설치를 멈추고 배포 담당자에게 다시 받으십시오.

### 3-2. 이미지 불러오기

```sh
docker load -i trid-onprem-<커밋>.tar
docker images | grep -E 'trid-onprem|caddy'
```

`trid-onprem-app:latest`, `trid-onprem-llm:latest`, `caddy:2.8.4` 세 개가 보여야 합니다.

### 3-3. 호스트 이름과 `.env`

사용자가 브라우저에 입력할 주소의 호스트 이름(예: `trid.example.internal`)을 정하고, 같은 폴더에 `.env` 파일을 만듭니다.

```sh
echo 'TRID_HOSTNAME=trid.example.internal' > .env
```

`TRID_HOSTNAME`은 인증서의 이름(SAN)과 일치해야 하고, 다음 절 설정 파일의 `hostname`과 정확히 같아야 합니다. 앱은 모든 POST 요청의 `Origin`이 `https://<hostname>`인지 확인하므로, 값이 다르면 로그인을 포함한 모든 POST가 "허용되지 않은 요청입니다."로 실패합니다. 접속 주소는 포트 없이 `https://<호스트 이름>/` 이어야 합니다.

선택: `.env`에 `LLM_THREADS=<숫자>`를 적으면 모델 서버 스레드 수를, `TRID_MAX_BODY=<크기>`를 적으면 Caddy의 요청 본문 상한을 바꿉니다(기본 3MB, 3-4절).

### 3-4. 설정 파일과 `data` 폴더

```sh
cp onprem.example.json data/onprem.json
chown -R 10001:10001 data     # 앱은 uid 10001로 실행됩니다
```

`data/onprem.json`을 열어 `hostname`을 `.env`의 `TRID_HOSTNAME`과 같게 고칩니다. 견본 내용은 다음과 같습니다.

```json
{
  "hostname": "trid.example.internal",
  "port": 8080,
  "data_dir": "/data",
  "llm_base": "http://llm:8080",
  "tokenizer_path": "/opt/trid/tokenizer.json",
  "retention_days": 365
}
```

- `hostname` 외 값은 보통 바꾸지 않습니다. `port`, `data_dir`, `llm_base`, `tokenizer_path`는 컨테이너 내부 값이므로 그대로 두십시오.
- `retention_days`: 이용 기록 보관 일수(기본 365, 1 이상). 4-4절 참조.
- 업로드 제한을 바꾸려면 `"pdf": {"max_bytes": 2097152, "max_pages": 20, "max_characters": 50000, "wall_seconds": 15.0}` 형태로 필요한 키만 추가합니다. 웹 화면은 `max_bytes` 값을 서버에서 받아 그 크기를 넘는 파일을 올리기 전에 막습니다. 단 Caddy가 요청 본문을 `TRID_MAX_BODY`(기본 3MB)로 제한하므로, `max_bytes`를 올릴 때는 `.env`에 `TRID_MAX_BODY=<max_bytes + 1MB 이상>`(예: `TRID_MAX_BODY=6MB`)도 함께 적고 `docker compose up -d`로 `tls`를 다시 만드십시오. 변경한 뒤 `docker compose restart app`도 필요합니다.
- 알 수 없는 키나 잘못된 형식이 있으면 서버가 시작되지 않고 오류가 표시됩니다.

### 3-5. 인증서

HTTPS 인증서는 `certs/` 폴더의 `cert.pem`(인증서, 필요하면 중간 인증서 포함)과 `key.pem`(개인 키)을 사용합니다.

**기관 인증서를 쓰는 경우(권장).** 두 파일을 `certs/`에 넣고 개인 키 권한을 맞춥니다.

```sh
chmod 600 certs/key.pem
chown 0:0 certs/key.pem       # root 소유, 모드 600
```

Caddy 컨테이너는 파일 접근 우회 권한 없이 root로 실행되므로, 키가 root 소유 모드 600이 아니면 읽지 못해 시작에 실패합니다.

**자체 서명 인증서를 쓰는 경우.** 시험 설치용입니다.

```sh
sudo ./make_self_signed.sh trid.example.internal certs
```

root나 `sudo`로 실행하면 키 소유자가 자동으로 `0:0`이 됩니다. 일반 사용자로 실행했다면 스크립트가 안내하는 대로 `sudo chown 0:0 certs/key.pem`을 실행하십시오. 유효기간은 825일입니다. 자체 서명 인증서는 브라우저가 "연결이 안전하지 않음" 경고를 표시하므로 직원에게 미리 알리거나, 기관 PC에 `certs/cert.pem`을 신뢰하는 인증서로 배포하십시오.

### 3-6. 시작

```sh
docker compose up -d
docker compose ps             # llm, app, tls 모두 running이어야 합니다
docker compose logs app       # "서버를 시작합니다"가 보이면 정상
```

`docker compose logs`에는 질문, 파일 이름, 답변, PDF 내용이 남지 않습니다(예외 클래스 이름만 기록).

### 3-7. 첫 관리자 만들기

```sh
docker compose exec app python -I -m trid.onprem create-admin --config /data/onprem.json --username <계정이름>
```

비밀번호를 두 번 묻습니다(최소 10자). 계정 이름은 영문 소문자, 숫자, `.`, `_`, `-`로 2~32자입니다. 이렇게 만든 첫 관리자는 비밀번호 변경을 강제하지 않습니다. 이후 계정은 웹 관리 화면에서 만들며, 그렇게 만든 계정과 비밀번호를 재설정한 계정은 첫 로그인 때 비밀번호를 바꿔야 합니다.

브라우저에서 `https://<호스트 이름>/`에 접속해 로그인되는지 확인합니다.

### 3-8. 배포 전 점검 (release_check.py)

설치가 끝나면 점검 도구로 실제 설치본을 한 번 돌려 봅니다. 이 도구는 파이썬 3 표준 라이브러리만 쓰므로 별도 설치가 필요 없으며, 서버에 접속할 수 있는 PC 어디서나 실행할 수 있습니다.

1. 가상 점검 자료 `release-document.txt`("가상 자료 · 실제 기관 규정 아님"으로 시작하는 새로 만든 가상 여비 규정)를 텍스트 PDF로 바꿉니다. 예: LibreOffice가 있는 PC에서

   ```sh
   soffice --headless --convert-to pdf release-document.txt
   ```

   만들어진 `release-document.pdf`가 2MiB 이하이고 텍스트 PDF(글자를 드래그해 복사할 수 있음)인지 확인합니다. 한글 글꼴이 없으면 글자가 깨지므로 한글 글꼴이 설치된 PC에서 변환하고, 변환된 PDF를 열어 한글이 깨지지 않았는지 반드시 눈으로 확인하십시오.
2. 점검을 실행합니다.

   ```sh
   python3 release_check.py --base https://trid.example.internal --user <관리자 계정> \
       --pdf release-document.pdf --questions release-questions.json --out report.json
   ```

   - 비밀번호는 실행 중에 묻습니다(명령줄에 쓰지 않습니다).
   - 인증서가 기관 CA 발급이면 그대로 검증합니다. 사설 CA나 자체 서명이면 `--cafile certs/cert.pem`으로 서버 인증서를 지정하거나, 시험 목적에 한해 `--insecure`로 검증을 건너뜁니다.
   - 계정이 비밀번호 변경 대기 상태면 "먼저 웹에서 비밀번호를 변경하세요"라는 메시지와 함께 중단합니다. 웹에서 한 번 로그인해 변경한 뒤 다시 실행하십시오.
   - 끝나면 스스로 로그아웃합니다.
3. `report.json`에는 질문별 `status`, 인용 수(`quotes`), 쪽 번호(`pages`), 처리 시간(`seconds`), 해설 유무(`has_commentary`)와 요약이 기록됩니다. 질문 내용과 답변 본문은 기록되지 않습니다. 이 파일은 점검을 실행한 PC에만 남습니다.
4. 도구는 기대값과 비교하지 않습니다. 사람이 웹에서 같은 PDF와 `release-questions.json`의 질문을 입력해 답을 읽고 판단합니다. 질문에는 문서에 답이 있는 것, 숫자를 묻는 것, 문서에 없는 것(`not_in_document`)이 섞여 있습니다. 없는 내용은 "찾지 못했다"로 나와야 하고, 숫자 질문은 인용된 원문과 대조해 확인합니다.

## 4. 운영

### 4-1. 접속과 세션

- 세션은 마지막 사용 후 30분이 지나면 만료됩니다. 만료 뒤 요청하면 "세션이 만료됐습니다. 페이지를 새로고침해 주세요." 또는 로그인 화면이 나옵니다.
- 동시 로그인 세션은 최대 32개입니다. 초과하면 "현재 접속 인원이 많습니다. 잠시 후 다시 시도해 주세요."가 표시됩니다.
- 비밀번호는 최소 10자입니다.

### 4-2. 계정 관리 (관리자)

관리자로 로그인하면 `https://<호스트 이름>/admin` 화면에서 다음을 할 수 있습니다.

- 계정 목록(역할, 잠금, 비밀번호 변경 대기 여부) 조회
- 계정 만들기: 역할은 `user`(일반)와 `admin`(관리자). 임시 비밀번호를 지정하며 첫 로그인 때 변경이 강제됩니다.
- 역할 변경, 계정 잠금/잠금 해제(관리자가 직접 잠그면 로그인할 수 없고, 잠금 해제로 풀며 실패 횟수 잠금도 함께 풀림), 비밀번호 재설정(재설정 후 첫 로그인 때 변경 강제)
- 로그인에 연속 5번 실패하면 계정이 15분 잠깁니다. 잠긴 계정으로 로그인하면 화면에는 일반 로그인 실패와 같은 "아이디 또는 비밀번호가 올바르지 않습니다."가 표시됩니다(계정 존재 여부를 숨기기 위함). 15분이 지나면 자동으로 풀리고, 급하면 관리자가 잠금 해제를 합니다.
- 비밀번호 변경이 필요한 계정은 변경 전까지 다른 기능을 쓸 수 없습니다. 관리 화면은 변경이 끝난 관리자만 열 수 있습니다.
- 관리자 계정은 한 명 이상 활성 상태여야 합니다. 모든 관리자가 잠기거나 비밀번호를 잊었다면 3-7절 명령으로 새 관리자를 만드십시오.

### 4-3. 이용 기록

관리 화면에서 기간, 계정, 이벤트 종류로 조회하고(한 번에 200건) CSV로 내려받을 수 있습니다. 이용 기록은 웹에서 수정하거나 삭제할 수 없습니다.

**기록하는 항목.** 모든 기록은 시각, 계정, 접속 IP, 이벤트, 결과와 아래의 부가 항목만 가집니다.

| 이벤트 | 시점 |
| --- | --- |
| `login_ok`, `login_failed`(비밀번호 오류, 잠금 계정이면 `error=disabled`), `login_locked`, `logout`, `session_expired` | 로그인, 실패, 잠김, 로그아웃, 세션 만료 |
| `password_changed` | 비밀번호 변경 |
| `pdf_upload` | PDF 올리기. 부가 항목: PDF 파일 전체의 `sha256`, `pages`(쪽 수), `status` |
| `ask` | 질문. 부가 항목: 처리 시간 `seconds`, 결과 `status`(`quoted` 원문 일치 인용 있음, `answer` 원문 위치 일치, `unverified` 인용 없이 해설만, `review` 검증 보류, `not_found` 근거 못 찾음, `clarify` 되묻기, `model_failure` 모델이 질문·지시문을 되풀이하는 등 응답 실패), 오류일 때 `error` 코드(`busy`, `context_overflow`, `bad_request`, `model_unavailable`) |
| `clear` | 올린 문서 지우기 |
| `admin_create_user`, `admin_update_user`, `admin_reset_password` | 관리자의 계정 작업. 부가 항목: 대상 계정 `target`, `action` |
| `admin_export` | CSV 내려받기. 부가 항목: 내보낸 건수 `count` |
| `retention_purge` | 보관 기간이 지난 기록 정리. 부가 항목: 정리한 건수 `count` |

**기록하지 않는 항목.** 다음은 이용 기록에도, 로그에도, 임시 파일에도 남기지 않습니다.

- 파일 이름
- 질문 내용
- 답변 내용(인용, 해설 포함)
- PDF 본문 텍스트
- 비밀번호(해시로만 저장)

`pdf_upload`의 `sha256`은 PDF 파일 바이트의 지문이며, 파일 내용을 되살릴 수 없습니다. 같은 파일이 올라왔는지만 확인하는 용도입니다. 올린 PDF와 추출 텍스트는 서버 메모리에만 있고 로그아웃·세션 만료·"자료 지우기" 때 사라집니다.

**CSV 형식.** 열은 `ts, username, ip, event, result, detail`이고 `ts`는 유닉스 시각(초)입니다. 파일은 UTF-8 BOM이 붙어 있어 엑셀에서 한글이 깨지지 않습니다. `=`, `+`, `-`, `@`, 탭, 줄바꿈 복귀로 시작하는 셀은 수식으로 실행되지 않도록 앞에 `'`가 붙습니다.

### 4-4. 보관 기간과 정리

- `data/onprem.json`의 `retention_days`(기본 365일, 1 이상)보다 오래된 이용 기록을 서버가 시작할 때와 이후 24시간마다 자동 삭제합니다. 정리 사실은 `retention_purge`로 기록됩니다.
- 수동 정리: `docker compose exec app python -I -m trid.onprem purge --config /data/onprem.json`
- 보관 기간을 바꾸면 `docker compose restart app`으로 반영합니다.
- 기관 규정상 더 오래 보관해야 하면 CSV를 정기적으로 내려받아 기관 방식대로 보관하십시오.

## 5. 백업·복구

백업 대상은 `data/` 폴더입니다. 계정과 이용 기록은 `data/onprem.db`(SQLite, 같은 폴더의 `onprem.db-wal`, `onprem.db-shm` 포함)에, 설정은 `data/onprem.json`에 있습니다. 인증서(`certs/`)와 `.env`도 함께 보관하십시오.

**방법 A. 서비스를 멈추고 복사(가장 단순).**

```sh
docker compose stop app
cp -a data data-backup-$(date +%Y%m%d)
docker compose start app
```

**방법 B. SQLite 백업.** 호스트에 `sqlite3`가 있을 때입니다. `data/`는 uid 10001 소유이고 WAL 모드라 일반 사용자는 열 수 없으므로 `sudo`가 필요하며, 앱이 쓰는 중에는 하지 말고 `docker compose stop app` 뒤에 실행하십시오.

```sh
docker compose stop app
sudo sqlite3 data/onprem.db ".backup 'backup-onprem.db'"
sudo cp data/onprem.json backup-onprem.json
docker compose start app
```

**복구.**

```sh
docker compose stop app
cp -a data-backup-<날짜>/. data/        # 방법 B로 받은 파일이면 backup-onprem.db를 data/onprem.db로 복사
rm -f data/onprem.db-wal data/onprem.db-shm   # 방법 B 파일을 복원할 때만
chown -R 10001:10001 data
docker compose start app
```

복구 뒤 관리자로 로그인해 계정 목록과 이용 기록을 확인합니다. 백업 파일에는 계정의 비밀번호 해시와 이용 기록(IP 포함)이 들어 있으므로 접근을 제한하십시오.

## 6. 업그레이드·되돌리기

이미지는 `trid-onprem-app:latest`, `trid-onprem-llm:latest`처럼 고정 태그 `latest`로 들어 있고 `compose.yaml`이 그 태그를 씁니다. 따라서 새 tar를 `docker load`하면 같은 태그가 새 이미지로 옮겨 가며, 이전 이미지는 태그 없는 상태로 남습니다. 되돌리려면 이전 tar를 다시 불러와야 하므로 **이전 버전의 폴더와 tar를 지우지 말고 보관**하십시오(폴더 이름 `trid-onprem-<커밋>`이 버전입니다).

**업그레이드.**

1. 5절 방법 A로 `data/`를 백업합니다.
2. 새 묶음을 반입하고 `sha256sum -c SHA256SUMS`로 확인합니다.
3. 운영 중인 폴더에서 `docker compose down`으로 서비스를 내립니다.
4. 새 폴더에 기존 `data/`, `certs/`, `.env`를 옮기거나 복사합니다(`data/onprem.json`은 새 `onprem.example.json`과 키를 비교).
5. 새 폴더에서 `docker load -i trid-onprem-<새 커밋>.tar` 후 `docker compose up -d`.
6. 로그인과 3-8절 점검을 다시 실행합니다.

**되돌리기.**

1. `docker compose down`
2. 이전 폴더에서 `docker load -i trid-onprem-<이전 커밋>.tar` (`latest` 태그가 이전 이미지로 돌아옵니다)
3. 필요하면 업그레이드 전에 받은 백업으로 `data/`를 복구(5절). 업그레이드가 데이터 구조를 바꿨는지 알 수 없으면 반드시 복구하십시오.
4. 이전 폴더에서 `docker compose up -d`

## 7. 인증서 교체

```sh
docker compose stop tls
cp <새 인증서> certs/cert.pem
cp <새 키> certs/key.pem
chmod 600 certs/key.pem
chown 0:0 certs/key.pem
docker compose up -d tls
```

호스트 이름이 바뀌면 `.env`의 `TRID_HOSTNAME`, `data/onprem.json`의 `hostname`, 인증서의 SAN을 모두 새 이름으로 맞춘 뒤 `docker compose up -d`(앱은 `docker compose restart app`)를 실행하십시오. 교체 후 브라우저 접속과 로그인을 확인합니다. 자체 서명 인증서는 만료일(825일)을 기록해 두십시오.

## 8. 문제 해결

| 증상 | 원인과 조치 |
| --- | --- |
| "모델 서버에 연결할 수 없습니다. 관리자에게 문의해 주세요." | `app`이 `llm`에 닿지 못했습니다. `docker compose ps`로 `llm` 상태를 보고 `docker compose logs llm`을 확인한 뒤 `docker compose restart llm`. 모델 서버는 시작 후 모델을 읽는 동안 잠시 응답하지 않을 수 있으니 1~2분 뒤 다시 시도하십시오. 이용 기록의 `ask` 오류 코드는 `model_unavailable`입니다. |
| 로그인이 계속 "아이디 또는 비밀번호가 올바르지 않습니다." | 비밀번호 오류이거나 계정이 5회 실패로 15분 잠긴 상태입니다. 관리자가 관리 화면에서 잠금 해제 또는 비밀번호 재설정을 합니다. 관리자 본인이 잠겼으면 15분을 기다리거나 다른 관리자가 해제합니다. |
| "세션이 만료됐습니다. 페이지를 새로고침해 주세요." | 30분 동안 사용하지 않아 세션이 끝났습니다. 새로고침 후 다시 로그인합니다. 올려 둔 PDF도 지워지므로 다시 올립니다. |
| 로그인을 포함한 모든 동작이 "허용되지 않은 요청입니다." | Origin 불일치입니다. 접속 주소의 호스트 이름이 `data/onprem.json`의 `hostname`과 다르거나, IP나 다른 이름으로 접속했거나, 포트를 붙였습니다. `.env`의 `TRID_HOSTNAME`, `hostname`, 인증서 SAN, 접속 주소를 모두 같은 이름으로 맞추고 `docker compose restart app` 하십시오. |
| "다른 질문을 처리 중입니다. 잠시 후 다시 시도해 주세요." | 한 번에 한 건만 처리합니다. 잠시 뒤 다시 시도합니다. 자주 나오면 2절의 사양을 점검하십시오. |
| "요청이 많습니다. 1분 뒤 다시 시도해 주세요." | 세션(로그인)당 분당 20회 제한입니다. |
| "현재 접속 인원이 많습니다." | 로그인 세션이 32개 찼습니다. 유휴 세션은 30분 뒤 정리됩니다. |
| PDF를 읽지 못함 | 화면에 사유가 표시됩니다. 스캔·이미지 PDF, 2MiB 초과, 20쪽 초과, 추출 글자 50,000자 초과, 암호 걸린 PDF는 읽지 못합니다. |
| `tls`가 시작하지 않고 인증서 오류 | `certs/cert.pem`, `certs/key.pem` 존재와 키 권한(root 소유 모드 600)을 확인합니다(3-5절). `docker compose logs tls`. |
| `app`이 시작하지 않음 | `docker compose logs app`. 설정 파일 키 오류, `data/` 쓰기 권한 문제(`chown -R 10001:10001 data`), 토크나이저 sha256 불일치 메시지가 표시됩니다. |
| `docker compose up`이 "set TRID_HOSTNAME in .env" 오류 | `.env`가 없거나 `TRID_HOSTNAME`이 비어 있습니다(3-3절). |

문의 시 `docker compose ps`, `docker compose logs --tail=100 <서비스>`의 출력을 첨부하십시오. 로그에는 질문, 파일 이름, 문서 내용이 없습니다.
