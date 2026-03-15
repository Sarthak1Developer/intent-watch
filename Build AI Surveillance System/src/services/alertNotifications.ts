import type { Alert } from './api';

const seenAlertKeys = new Set<string>();

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

export function shouldNotifyAlert(a: Alert): boolean {
  const sev = String(a.severity ?? '').toLowerCase();
  if (sev === 'critical' || sev === 'high') return true;
  const t = (a.type || '').toLowerCase();
  return t.includes('weapon') || t.includes('zone') || t.includes('bag');
}

export function notifyAlert(a: Alert): void {
  try {
    if (typeof window === 'undefined') return;
    if (!('Notification' in window)) return;
    if (Notification.permission !== 'granted') return;
    if (!shouldNotifyAlert(a)) return;

    const key = getAlertKey(a);
    if (seenAlertKeys.has(key)) return;
    seenAlertKeys.add(key);

    new Notification(a.type, {
      body: a.message,
      tag: key,
    });
  } catch {
    // ignore
  }
}
