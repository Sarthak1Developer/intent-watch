// API Service for IntentWatch Backend Integration

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface Alert {
  id?: string;
  type: string;
  message: string;
  time: string;
  timestamp?: string;
  severity?: string | null;
  camera?: string | null;
}

export interface AnalyticsData {
  total: number;
  counts: { [key: string]: number };
  severity?: { [key: string]: number };
  by_day?: Array<{ date: string; day: string; alerts: number }>;
  by_hour?: Array<{ hour: string; alerts: number }>;
  threat_trends?: Array<{ date: string; day: string; Running: number; Loitering: number; 'Unattended Bag': number }>;
  recent: Alert[];
}

// Video API
export const videoAPI = {
  // Upload video file
  async uploadVideo(file: File): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${API_BASE_URL}/video/upload`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      throw new Error('Failed to upload video');
    }
    
    return response.json();
  },

  // Start video stream
  async startVideo(source: string): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/video/start`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ source }),
    });
    
    if (!response.ok) {
      throw new Error('Failed to start video');
    }
    
    return response.json();
  },

  // Stop video stream
  async stopVideo(options?: { keepalive?: boolean }): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/video/stop`, {
      method: 'POST',
      keepalive: Boolean(options?.keepalive),
    });
    
    if (!response.ok) {
      throw new Error('Failed to stop video');
    }
    
    return response.json();
  },

  // Get video stream URL
  getStreamUrl(): string {
    return `${API_BASE_URL}/video/stream`;
  },

  getStreamUrlById(streamId: string): string {
    return `${API_BASE_URL}/video/stream/${encodeURIComponent(streamId)}`;
  },

  // Start webcam
  async startWebcam(deviceId: number = 0): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/video/start-camera`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ device_id: deviceId }),
    });
    
    if (!response.ok) {
      throw new Error('Failed to start webcam');
    }
    
    return response.json();
  },

  // Get status
  async getStatus(): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/video/status`);
    
    if (!response.ok) {
      throw new Error('Failed to get status');
    }
    
    return response.json();
  },

  async listStreams(): Promise<{ streams: Array<{ id: string; mode: string | null; path: any; running: boolean }> }> {
    const response = await fetch(`${API_BASE_URL}/video/streams`, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error('Failed to list streams');
    }
    return response.json();
  },

  async startStream(streamId: string, source: string): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/video/streams/start`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ stream_id: streamId, source }),
    });

    if (!response.ok) {
      let detail = '';
      try {
        const data = await response.json();
        detail = String((data as any)?.detail ?? '');
      } catch {
        // ignore
      }
      throw new Error(detail || 'Failed to start stream');
    }
    return response.json();
  },

  async stopStream(streamId: string, options?: { keepalive?: boolean }): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/video/streams/stop`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      keepalive: Boolean(options?.keepalive),
      body: JSON.stringify({ stream_id: streamId }),
    });

    if (!response.ok) {
      let detail = '';
      try {
        const data = await response.json();
        detail = String((data as any)?.detail ?? '');
      } catch {
        // ignore
      }
      throw new Error(detail || 'Failed to stop stream');
    }
    return response.json();
  },

  // Set normalized zones (0..1) for backend processing
  async setZones(zones: Array<{ id: string; name: string; severity: string; x: number; y: number; width: number; height: number }>): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/video/zones`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ zones }),
    });

    if (!response.ok) {
      throw new Error('Failed to set zones');
    }

    return response.json();
  },

  async setZonesForStream(streamId: string, zones: Array<{ id: string; name: string; severity: string; x: number; y: number; width: number; height: number }>): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/video/zones/${encodeURIComponent(streamId)}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ zones }),
    });

    if (!response.ok) {
      throw new Error('Failed to set zones');
    }

    return response.json();
  },
};

// Alerts API
export const alertsAPI = {
  // Get live alerts
  async getLiveAlerts(): Promise<Alert[]> {
    const response = await fetch(`${API_BASE_URL}/alerts/live`, { cache: 'no-store' });
    
    if (!response.ok) {
      throw new Error('Failed to fetch alerts');
    }
    
    return response.json();
  },

  // Get analytics data
  async getAnalytics(): Promise<AnalyticsData> {
    const response = await fetch(`${API_BASE_URL}/alerts/analytics`, { cache: 'no-store' });
    
    if (!response.ok) {
      throw new Error('Failed to fetch analytics');
    }
    
    return response.json();
  },

  // Clear all alerts
  async clearAlerts(): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/alerts/clear`, {
      method: 'POST',
    });
    
    if (!response.ok) {
      throw new Error('Failed to clear alerts');
    }
    
    return response.json();
  },
};

// System API
export const systemAPI = {
  // Check backend health
  async checkHealth(): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/`);
    
    if (!response.ok) {
      throw new Error('Backend is not responding');
    }
    
    return response.json();
  },

  async getMetrics(): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/metrics`, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error('Failed to fetch metrics');
    }
    return response.json();
  },
};

export default {
  videoAPI,
  alertsAPI,
  systemAPI,
};
