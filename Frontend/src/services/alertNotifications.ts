import type { Alert } from './api';

const seenAlertKeys = new Set<string>();

const SEEN_STORAGE_KEY = 'intentwatch.notifications.seenKeys.v1';
const MAX_SEEN_KEYS = 400;

let _seenLoaded = false;
let _seenQueue: string[] = [];

let _audioCtx: AudioContext | null = null;

function _isSoundEnabled(): boolean {
  try {
    const raw = window.localStorage.getItem('intentwatch.settings.v1');
    if (!raw) return true;
    const parsed = JSON.parse(raw) as any;
    if (parsed && typeof parsed.sound === 'boolean') return parsed.sound;
    return true;
  } catch {
    return true;
  }
}

function _beepBestEffort(): void {
  try {
    if (typeof window === 'undefined') return;
    if (!_isSoundEnabled()) return;

    const AnyWindow = window as any;
    const Ctx = AnyWindow.AudioContext || AnyWindow.webkitAudioContext;
    if (!Ctx) return;

    if (_audioCtx == null) {
      _audioCtx = new Ctx();
    }

    // Some browsers start audio contexts suspended until user interaction.
    if (_audioCtx.state === 'suspended') {
      void _audioCtx.resume().catch(() => undefined);
    }

    const ctx = _audioCtx;
    if (!ctx) return;

    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = 'sine';
    osc.frequency.value = 880;

    const now = ctx.currentTime;
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(0.15, now + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.18);

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.start(now);
    osc.stop(now + 0.2);
  } catch {
    // ignore
  }
}

export function ensureNotificationPermissionNonBlocking(): void {
  try {
    if (typeof window === 'undefined') return;
    if (!('Notification' in window)) return;
    if (Notification.permission === 'granted') return;
    if (Notification.permission === 'denied') return;
    // Do not await; never block UI on permission prompt.
    void Notification.requestPermission();
  } catch {
    // ignore
  }
}

function getAlertKey(a: Alert): string {
  return String(a.id ?? a.timestamp ?? `${a.type}-${a.time}-${a.message}`);
}

function _loadSeenOnce(): void {
  try {
    if (_seenLoaded) return;
    _seenLoaded = true;
    if (typeof window === 'undefined') return;

    const raw = window.localStorage.getItem(SEEN_STORAGE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return;

    const keys = parsed
      .map((k) => String(k ?? '').trim())
      .filter((k) => k);

    _seenQueue = keys.slice(-MAX_SEEN_KEYS);
    for (const k of _seenQueue) seenAlertKeys.add(k);
  } catch {
    // ignore
  }
}

function _persistSeenBestEffort(): void {
  try {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem(SEEN_STORAGE_KEY, JSON.stringify(_seenQueue.slice(-MAX_SEEN_KEYS)));
  } catch {
    // ignore
  }
}

export function shouldNotifyAlert(a: Alert): boolean {
  const t = (a.type || '').toLowerCase();
  // Only these types should trigger OS/device notifications.
  // - Weapon detection
  // - Unattended bag
  // - Zone alerts (restricted + unrestricted)
  if (t.includes('weapon') || t.includes('bag')) return true;

  if (t.includes('zone')) {
    return true;
  }

  return false;
}

export function notifyAlert(a: Alert): void {
  try {
    if (typeof window === 'undefined') return;
    if (!('Notification' in window)) return;
    if (Notification.permission !== 'granted') return;
    if (!shouldNotifyAlert(a)) return;

    _loadSeenOnce();

    const key = getAlertKey(a);
    if (seenAlertKeys.has(key)) return;
    seenAlertKeys.add(key);

    _seenQueue.push(key);
    if (_seenQueue.length > MAX_SEEN_KEYS) {
      _seenQueue = _seenQueue.slice(-MAX_SEEN_KEYS);
      // Rebuild set occasionally to keep it bounded.
      seenAlertKeys.clear();
      for (const k of _seenQueue) seenAlertKeys.add(k);
    }
    _persistSeenBestEffort();

    const n = new Notification(a.type, {
      body: a.message,
      tag: key,
    });

    // Best-effort: clicking the notification should focus the app tab.
    try {
      n.onclick = () => {
        try {
          window.focus();
        } catch {
          // ignore
        }
      };
    } catch {
      // ignore
    }

    // Best-effort audible cue (only when the web app is open).
    _beepBestEffort();
  } catch {
    // ignore
  }
}
