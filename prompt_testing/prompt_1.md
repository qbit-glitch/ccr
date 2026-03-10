# One-Shot Build Prompt: Cross-Platform AI Assistant in Rust

> Copy everything below the line and paste it into Claude Code, Cursor, or any AI coding agent.

---

## PROMPT START

You are building a cross-platform AI-native assistant called **Vox** from scratch. The project uses a **shared Rust core** with thin native platform shells. Build the entire project in one pass — compilable, runnable, and well-structured.

---

### 1. PROJECT OVERVIEW

**Vox** is a background AI assistant activated by a single volume-down press (Android) or a global hotkey (desktop). It provides concise 2-3 line answers, extracts TODOs, schedules calendar events, sets reminders, and can initiate phone/WhatsApp calls. The UI is a translucent glass overlay that appears on-screen.

---

### 2. ARCHITECTURE

```
vox/
├── Cargo.toml                    # Workspace root
├── crates/
│   ├── vox-core/                 # Shared Rust library (ALL business logic)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs            # Public API surface exposed via UniFFI
│   │       ├── ai/
│   │       │   ├── mod.rs
│   │       │   ├── client.rs     # HTTP client for Claude API (reqwest + streaming)
│   │       │   ├── prompt.rs     # System prompts, response formatting, intent detection
│   │       │   └── models.rs     # Request/Response types
│   │       ├── assistant/
│   │       │   ├── mod.rs
│   │       │   ├── intent.rs     # Intent parser: detect TODO, calendar, reminder, call, general query
│   │       │   ├── responder.rs  # Orchestrator: takes user input → routes to correct handler → returns response
│   │       │   └── context.rs    # Conversation history manager (sliding window, max 20 turns)
│   │       ├── todo/
│   │       │   ├── mod.rs
│   │       │   ├── store.rs      # CRUD operations on TODOs (SQLite via rusqlite)
│   │       │   ├── extractor.rs  # AI-powered TODO extraction from conversation
│   │       │   └── models.rs     # Todo { id, title, body, priority, done, created_at, due_date }
│   │       ├── calendar/
│   │       │   ├── mod.rs
│   │       │   ├── scheduler.rs  # Create/read/update/delete calendar events
│   │       │   ├── reminder.rs   # Reminder scheduling with platform notification bridge
│   │       │   └── models.rs     # CalendarEvent { id, title, description, start, end, reminder_minutes }
│   │       ├── calls/
│   │       │   ├── mod.rs
│   │       │   └── dialer.rs     # Platform-agnostic call intent (SIM call, WhatsApp deeplink)
│   │       ├── audio/
│   │       │   ├── mod.rs
│   │       │   ├── recorder.rs   # Audio capture abstraction (platform provides raw PCM)
│   │       │   ├── stt.rs        # Speech-to-text: Whisper API client (cloud) or whisper.cpp bindings (local)
│   │       │   └── tts.rs        # Text-to-speech: platform TTS bridge or cloud TTS
│   │       ├── storage/
│   │       │   ├── mod.rs
│   │       │   └── db.rs         # SQLite database initialization, migrations, connection pool
│   │       └── bridge.rs         # UniFFI interface definition: all functions exposed to Kotlin/Swift/JS
│   │
│   └── vox-uniffi/               # UniFFI binding generator crate
│       ├── Cargo.toml
│       ├── src/
│       │   └── lib.rs            # Re-exports vox-core with UniFFI scaffolding
│       └── src/vox.udl           # UniFFI definition file
│
├── android/                      # Android shell (Kotlin)
│   ├── app/
│   │   ├── build.gradle.kts
│   │   └── src/main/
│   │       ├── AndroidManifest.xml
│   │       ├── java/com/vox/app/
│   │       │   ├── VoxApplication.kt          # App init, load Rust .so
│   │       │   ├── VoxAccessibilityService.kt  # Volume button interception
│   │       │   ├── VoxOverlayService.kt        # Floating glass UI overlay
│   │       │   ├── ui/
│   │       │   │   ├── OverlayView.kt          # Glass morphism overlay (Jetpack Compose)
│   │       │   │   ├── ChatBubble.kt           # Message bubbles
│   │       │   │   └── GlassTheme.kt           # Translucent theme definitions
│   │       │   ├── audio/
│   │       │   │   └── AudioCaptureManager.kt  # Mic recording, feeds PCM to Rust
│   │       │   └── platform/
│   │       │       ├── CalendarBridge.kt        # Android CalendarProvider integration
│   │       │       ├── NotificationManager.kt   # Reminder notifications via AlarmManager
│   │       │       └── CallBridge.kt            # ACTION_CALL intent, WhatsApp deeplink
│   │       └── res/
│   │           └── layout/                      # Minimal XML layouts if needed
│   └── gradle/
│
├── desktop/                      # Desktop shell (Tauri v2)
│   ├── src-tauri/
│   │   ├── Cargo.toml            # Depends on vox-core directly (no UniFFI needed)
│   │   ├── src/
│   │   │   ├── main.rs           # Tauri app entry
│   │   │   ├── commands.rs       # Tauri commands wrapping vox-core functions
│   │   │   ├── hotkey.rs         # Global hotkey registration (Ctrl+Space or customizable)
│   │   │   ├── tray.rs           # System tray icon and menu
│   │   │   └── audio.rs          # Desktop audio capture (cpal crate)
│   │   └── tauri.conf.json
│   └── src/                      # Frontend (HTML/CSS/JS or React)
│       ├── index.html
│       ├── app.js                # Chat UI, glass morphism CSS, Tauri invoke calls
│       └── styles.css            # Glass UI styles
│
├── ios/                          # iOS shell (Swift) — future
│   └── Vox/
│       ├── VoxApp.swift
│       ├── RustBridge.swift       # C-FFI bridge to vox-core
│       └── OverlayView.swift      # SwiftUI glass overlay
│
└── scripts/
    ├── build-android.sh          # cargo ndk + gradle build
    ├── build-desktop.sh          # cargo tauri build
    └── setup.sh                  # Install dependencies (cargo, ndk, tauri-cli)
```

---

### 3. DETAILED SPECIFICATIONS

#### 3.1 Rust Core (`vox-core`)

**Cargo.toml dependencies:**
```toml
[dependencies]
reqwest = { version = "0.12", features = ["json", "stream", "rustls-tls"] }
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
rusqlite = { version = "0.31", features = ["bundled"] }
chrono = { version = "0.4", features = ["serde"] }
uuid = { version = "1", features = ["v4"] }
thiserror = "1"
tracing = "0.1"
uniffi = "0.28"
```

**AI Client (`ai/client.rs`):**
- Use the Anthropic Messages API (`POST https://api.anthropic.com/v1/messages`)
- Model: `claude-sonnet-4-20250514`
- System prompt enforces: "You are Vox, a concise AI assistant. Always respond in 2-3 short lines max. If a URL/link would help, include ONE relevant link. Detect user intents: if the user wants to save a TODO, schedule an event, set a reminder, or make a call, respond with a structured JSON block wrapped in <vox_action>...</vox_action> tags followed by a short confirmation message."
- Support streaming responses via SSE for real-time text display
- API key passed from platform shell at init, stored in-memory only

**Intent Detection (`assistant/intent.rs`):**
Parse AI responses for `<vox_action>` blocks. Supported intents:
```rust
enum VoxIntent {
    GeneralResponse { text: String, links: Vec<String> },
    SaveTodo { title: String, body: Option<String>, priority: Priority, due_date: Option<NaiveDate> },
    ScheduleEvent { title: String, description: Option<String>, start: DateTime<Utc>, end: DateTime<Utc>, reminder_minutes: Option<i32> },
    SetReminder { message: String, trigger_at: DateTime<Utc> },
    MakeCall { contact: String, method: CallMethod },  // CallMethod::Sim or CallMethod::WhatsApp
}
```

**TODO Store (`todo/store.rs`):**
- SQLite table: `todos(id TEXT PRIMARY KEY, title TEXT, body TEXT, priority INTEGER, done BOOLEAN, created_at TEXT, due_date TEXT)`
- Methods: `add_todo()`, `list_todos()`, `complete_todo()`, `delete_todo()`
- AI can extract TODOs automatically from conversation when it detects actionable items

**Calendar (`calendar/scheduler.rs`):**
- SQLite table: `events(id TEXT PRIMARY KEY, title TEXT, description TEXT, start_time TEXT, end_time TEXT, reminder_minutes INTEGER, created_at TEXT)`
- On Android: also sync to system calendar via CalendarBridge
- On Desktop: use local DB only, fire OS notifications for reminders

**Calls (`calls/dialer.rs`):**
- Expose `initiate_call(contact: String, method: CallMethod) -> CallRequest`
- Platform shell handles the actual call:
  - Android SIM: `Intent(Intent.ACTION_CALL, Uri.parse("tel:$number"))`
  - WhatsApp: `Intent(Intent.ACTION_VIEW, Uri.parse("https://wa.me/$number"))`
  - Desktop: open `tel:` URI or WhatsApp web link

**Audio (`audio/`):**
- `recorder.rs`: Accept raw PCM f32 samples from platform, buffer into chunks
- `stt.rs`: Send audio to OpenAI Whisper API (`POST https://api.openai.com/v1/audio/transcriptions`) or bind to whisper.cpp for local inference
- `tts.rs`: Platform delegates (Android TextToSpeech, macOS `say`, or cloud TTS)

**UniFFI Bridge (`bridge.rs` + `vox.udl`):**
Expose these functions to Kotlin/Swift:
```
namespace vox {
    // Lifecycle
    fn initialize(db_path: string, api_key: string);
    fn shutdown();

    // Chat
    fn send_message(text: string) -> VoxResponse;
    fn send_audio(pcm_data: sequence<float>) -> VoxResponse;

    // TODOs
    fn list_todos() -> sequence<Todo>;
    fn complete_todo(id: string) -> bool;
    fn delete_todo(id: string) -> bool;

    // Calendar
    fn list_events(from_date: string, to_date: string) -> sequence<CalendarEvent>;
    fn delete_event(id: string) -> bool;

    // State
    fn get_conversation_history() -> sequence<ChatMessage>;
    fn clear_conversation();
};
```

#### 3.2 Android Shell

**Volume Button Interception (`VoxAccessibilityService.kt`):**
```
AccessibilityService that:
1. Overrides onKeyEvent() to capture KEYCODE_VOLUME_DOWN
2. On single press: starts a 400ms timer
   - If no second press within 400ms → activate Vox overlay (consume the event)
   - If second press within 400ms → pass through as normal volume down (don't consume)
3. When activated: start VoxOverlayService, begin audio recording
4. Requires: BIND_ACCESSIBILITY_SERVICE permission, user must enable in Settings > Accessibility
```

**Glass Overlay UI (`VoxOverlayService.kt` + Jetpack Compose):**
```
- Window type: TYPE_APPLICATION_OVERLAY (requires SYSTEM_ALERT_WINDOW permission)
- Glass morphism effect:
  - Background: semi-transparent white/dark with blur (RenderEffect.createBlurEffect on Android 12+)
  - Border: 1px subtle white/gray border with rounded corners (24dp)
  - Shadow: soft elevation shadow
- Layout:
  - Compact floating card (width: 85% screen, max-height: 60% screen)
  - Scrollable chat area with user/assistant message bubbles
  - Bottom: text input field with mic button and send button
  - Top-right: close (X) button, pin button
  - Animated entry: slide up from bottom with fade-in
- Colors:
  - Light mode: rgba(255, 255, 255, 0.72) background, dark text
  - Dark mode: rgba(30, 30, 30, 0.78) background, light text
  - Accent: adaptive system accent color
```

**Required Android Permissions:**
```xml
<uses-permission android:name="android.permission.RECORD_AUDIO" />
<uses-permission android:name="android.permission.SYSTEM_ALERT_WINDOW" />
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.READ_CALENDAR" />
<uses-permission android:name="android.permission.WRITE_CALENDAR" />
<uses-permission android:name="android.permission.CALL_PHONE" />
<uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
<uses-permission android:name="android.permission.POST_NOTIFICATIONS" />
```

#### 3.3 Desktop Shell (Tauri v2)

**Global Hotkey (`hotkey.rs`):**
- Default: `Ctrl+Space` (configurable)
- On press: toggle overlay window visibility
- Use tauri-plugin-global-shortcut

**Glass UI (`src/styles.css`):**
```css
/* Glass morphism */
.vox-overlay {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.25);
    border-radius: 16px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
}

/* Dark mode */
@media (prefers-color-scheme: dark) {
    .vox-overlay {
        background: rgba(20, 20, 20, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
}
```

**Tauri Window Config:**
```json
{
    "label": "vox-overlay",
    "title": "Vox",
    "width": 420,
    "height": 520,
    "resizable": false,
    "decorations": false,
    "transparent": true,
    "alwaysOnTop": true,
    "visible": false,
    "center": true
}
```

---

### 4. IMPLEMENTATION RULES

1. **Rust style:** Use modern idiomatic Rust. Prefer `&value` on the value side (e.g., `if let Some(node) = &root`), never `ref` in patterns. Use `thiserror` for error types. Use `tracing` for logging.

2. **Error handling:** Define a `VoxError` enum in `vox-core` with variants for `ApiError`, `DatabaseError`, `AudioError`, `ParseError`, `NetworkError`. All public functions return `Result<T, VoxError>`.

3. **Async:** The Rust core uses `tokio` internally. UniFFI bridge functions block on the tokio runtime (create a `Runtime` at init). Tauri calls can be async natively.

4. **Database:** Initialize SQLite with WAL mode. Run migrations on first launch. Use prepared statements.

5. **Streaming:** For the desktop Tauri shell, implement SSE streaming from Claude API so text appears word-by-word in the UI. For Android via UniFFI, use a callback interface for streaming chunks.

6. **Security:** Never persist API keys to disk. Accept them at runtime (from Android SharedPreferences encrypted or desktop keychain). Sanitize all user input before SQL queries (use parameterized queries only).

7. **Testing:** Include unit tests for:
   - Intent parsing (test each VoxIntent variant)
   - TODO CRUD operations
   - Calendar event creation and querying
   - Conversation context window management

---

### 5. BUILD INSTRUCTIONS

Generate these working build scripts:

**`scripts/setup.sh`:**
```bash
#!/bin/bash
# Install Rust targets for Android
rustup target add aarch64-linux-android armv7-linux-androideabi x86_64-linux-android
# Install cargo-ndk for Android builds
cargo install cargo-ndk
# Install Tauri CLI for desktop builds
cargo install tauri-cli
# Install UniFFI bindgen
cargo install uniffi-bindgen-cli
```

**`scripts/build-android.sh`:**
```bash
#!/bin/bash
cd crates/vox-uniffi
cargo ndk -t arm64-v8a -t armeabi-v7a -t x86_64 -o ../../android/app/src/main/jniLibs build --release
uniffi-bindgen generate src/vox.udl --language kotlin --out-dir ../../android/app/src/main/java
cd ../../android && ./gradlew assembleRelease
```

**`scripts/build-desktop.sh`:**
```bash
#!/bin/bash
cd desktop && cargo tauri build
```

---

### 6. FIRST RUN FLOW

When the user first opens Vox:
1. Show a minimal onboarding screen: "Enter your Anthropic API key" with a text field
2. Store the key securely (Android EncryptedSharedPreferences / desktop keychain)
3. On Android: prompt to enable the AccessibilityService with a deep link to Settings
4. On Android: request SYSTEM_ALERT_WINDOW permission
5. Show a test message: user says "Hello" → Vox responds with "Hey! I'm Vox, your AI assistant. Try asking me anything, say 'remind me to...', or 'add a todo for...'"

---

### 7. DELIVERABLES

Build ALL of the following, fully implemented and compilable:

1. Complete `vox-core` Rust crate with all modules
2. UniFFI definitions and generated bindings
3. Android app with AccessibilityService, overlay UI (Jetpack Compose), and all platform bridges
4. Tauri desktop app with glass UI, global hotkey, and full chat functionality
5. SQLite migrations for todos and calendar tables
6. Unit tests for core logic
7. Build scripts that work

Write every file. Do not use placeholder comments like `// TODO: implement` — write the actual implementation. If a function needs 50 lines, write all 50 lines.

## PROMPT END