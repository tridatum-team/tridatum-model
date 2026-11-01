# SITE-MIGRATION-021 — 맥미니 정적 배경화면 전환 후 이관 검증 (사용자 승인)

**상태: 차단. 첫 단계(배경화면 정적 전환)가 macOS 권한 동의 화면에서 멈췄다.**
- 권한 화면을 우회하지 않았다. 배경화면은 바뀌지 않았다(원래 설정 그대로).
- 그래서 이관 관찰, 터널 전환, 맥북 정지는 하지 않았다. 원점은 맥북 그대로다.
- SITE-CONFIG-AUDIT-020의 원인(LaunchAgent 등록 → 알림 디스플레이 깨움 → Aerial 동영상)은 **시각 상관에 근거한 후보**다. 확정이 아니다.

## 1. 현재 배경화면 (읽기 전용 확인)

- 저장소 `~/Library/Application Support/com.apple.wallpaper/Store/Index.plist`: `AllSpacesAndDisplays`와 `SystemDefault` 모두 `Provider = "default"`(LastSet 2026-01-30).
- macOS 26.3.1 기본 배경화면은 동영상 "Tahoe Day"(`/System/Library/Desktop Pictures/.wallpapers/Tahoe Day/Tahoe Day.mov`, 170MB)다.
  - 재생 주체는 `WallpaperAerialsExtension`(pid 774). SITE-CONFIG-AUDIT-020 로그와 맞는다.
- 같은 장면의 정지 이미지: `/System/Library/ExtensionKit/Extensions/NeptuneOneWallpaper.appex/Contents/Resources/TahoeLight.heic`(24.6MB, OS 내장, Day에 해당하는 Light).
  - `Tahoe Day/thumbnail.heic`(38KB)는 해상도가 낮아 쓰지 않았다.

## 2. 시도와 결과

1. **원설정 보존:** `Index.plist`를 `~/Services/trid-site/wallpaper-backup/Index.plist.before-021`로 복사했다(sha256 `fd3aa55b…`, 원본과 같음).
2. **정적 전환 1회 시도 (19:21:27):** `osascript -e 'tell application "System Events" to tell every desktop to set picture to POSIX file ".../TahoeLight.heic"'` (보통 경로: System Events Apple Event).
   - 30초 안에 끝나지 않았다.
   - tccd 기록: 요청 주체 `com.apple.osascript`(pid 5280), 책임 프로세스 `com.apple.sshd-keygen-wrapper`(SSH 세션)에 대한 권한 요청(`AUTHREQ`).
   - 19:21:28.651에 `UserNotificationCenter`가 동작했다.
   - **맥미니 화면에 "sshd-keygen-wrapper(터미널/SSH)가 System Events를 제어하려 함" 류의 자동화 동의 대화상자가 떴을 가능성이 높다** [추정: 화면을 직접 보지 못함].
3. **우회 없음:**
   - 다른 경로는 쓰지 않았다: plist 직접 편집, WallpaperAgent 재시작, TCC DB 수정.
   - 대기 중이던 본인 `osascript`(pid 5280)는 19:24에 TERM으로 끝냈다. 나중에 동의 버튼이 눌려 의도하지 않은 시점(예: 시험 중)에 배경화면이 바뀌는 일을 막으려는 것이다.
4. **확인:** `Index.plist`는 여전히 `Provider = "default"`, sha 그대로. 압력 1. 기존 서비스 기준 그대로(8000/8501/8502=200, 18765/8766=403).

## 3. 진행에 필요한 사용자 조치 (둘 중 하나)

**A. 맥미니 화면에서 직접 바꾸기 (권장, 권한 부여 없음)**
1. 시스템 설정 → 배경화면.
2. 정지 이미지를 고른다. 예: "Tahoe" 정지 이미지, 또는 동영상이 아닌 사진·색상.
3. 바뀐 뒤 A가 `Index.plist` Provider와 `WallpaperAerialsExtension` 재생 로그로 정적 여부를 확인한다.

**B. 자동화 권한을 주고 A가 다시 1회 실행**
- 맥미니 화면에 대화상자가 남아 있으면 "허용"을 누르거나, 시스템 설정 → 개인정보 보호 및 보안 → 자동화에서 SSH(sshd-keygen-wrapper) → System Events를 켠다.
- 이것은 **SSH 세션이 System Events를 제어하는 넓은 권한**이다. 필요 없으면 A안을 권한다.
- 대화상자가 남아 있으면 "허용 안 함"을 눌러도 된다. A안으로 진행하면 된다.

**복구 자료:** 원래 배경화면으로 되돌리려면 시스템 설정 → 배경화면에서 "Tahoe Day"(기본 동영상)를 다시 고르면 된다. 백업 `Index.plist.before-021`은 기록용이며, A는 이 파일을 직접 덮어쓰지 않는다.

## 4. 하지 않은 것

- 다른 시스템 설정(알림, 디스플레이, 화면 보호기) 변경.
- 이관 관찰 시험(launchd CPU 적용 확인, 15분 시나리오), named tunnel 전환, 맥북 정지, 외부 검증.
  - 정적 전환이 되지 않은 상태에서 시험하면 SITE-MIGRATION-019 조건을 그대로 반복하게 되어 하지 않았다.
- 문서(docs) 갱신. 이관 전이다.

## 5. 정적 전환 후 이어서 할 일 (승인된 019·021 계획 그대로)

1. 정적 여부 확인: Index.plist Provider. 디스플레이를 깨우지 않는 비파괴 방법으로, 이후 알림이 와도 `WallpaperAerialsExtension` 재생 로그와 `VTDecoderXPCService` 기동이 없는지 관찰한다.
2. 운영 launchd 기동 직후 llama-server의 두 env 값만(`LLAMA_ARG_DEVICE`, `LLAMA_ARG_N_GPU_LAYERS`), 실행 인자, vmmap(IOAccelerator·GGUF 매핑)을 기록한다.
3. 019 가드 그대로 약 15분 관찰한다: 대화, 기관 전환, 문서 첨부, 독해기, 새 대화, 버전 목록, 삭제 ID 거부.
4. 통과하면:
   1. 맥북 터널 bootout → 맥미니 단일 connector.
   2. 맥북 TRID만 정지하고 자동 부활을 disable한다.
   3. 외부 검증을 하고 총괄에 알린다.
   4. docs를 실제 host·CPU·한계로 갱신한다.

SUPERVISION_REPORT_READY SITE-MIGRATION-021
