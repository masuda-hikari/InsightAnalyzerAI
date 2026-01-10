// InsightAnalyzerAI Service Worker v1.0
const CACHE_NAME = 'insightanalyzerai-v1';
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/guide/',
  '/guide/index.html',
  '/api/',
  '/api/index.html',
  '/assets/styles.min.css',
  '/assets/icon.svg',
  '/assets/icon-192.png',
  '/assets/icon-512.png',
  '/manifest.json'
];

// インストール時にキャッシュ
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('[SW] キャッシュを開きました');
        return cache.addAll(STATIC_ASSETS);
      })
      .then(() => self.skipWaiting())
      .catch((err) => console.log('[SW] キャッシュ追加エラー:', err))
  );
});

// アクティベート時に古いキャッシュを削除
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames
            .filter((name) => name !== CACHE_NAME)
            .map((name) => {
              console.log('[SW] 古いキャッシュを削除:', name);
              return caches.delete(name);
            })
        );
      })
      .then(() => self.clients.claim())
  );
});

// フェッチ時: ネットワーク優先、失敗時キャッシュ
self.addEventListener('fetch', (event) => {
  // POSTリクエストはキャッシュしない
  if (event.request.method !== 'GET') {
    return;
  }

  // 外部リソース（Google Fonts等）はネットワークのみ
  if (!event.request.url.startsWith(self.location.origin)) {
    return;
  }

  event.respondWith(
    fetch(event.request)
      .then((response) => {
        // 成功したレスポンスをキャッシュに保存
        if (response.status === 200) {
          const responseClone = response.clone();
          caches.open(CACHE_NAME)
            .then((cache) => cache.put(event.request, responseClone));
        }
        return response;
      })
      .catch(() => {
        // ネットワーク失敗時はキャッシュから返す
        return caches.match(event.request)
          .then((cachedResponse) => {
            if (cachedResponse) {
              return cachedResponse;
            }
            // HTMLページの場合はオフラインページを返す
            if (event.request.headers.get('accept').includes('text/html')) {
              return caches.match('/');
            }
            return new Response('オフラインです', {
              status: 503,
              statusText: 'Service Unavailable'
            });
          });
      })
  );
});

// プッシュ通知（将来用）
self.addEventListener('push', (event) => {
  if (event.data) {
    const data = event.data.json();
    const options = {
      body: data.body || '新しい通知があります',
      icon: '/assets/icon-192.png',
      badge: '/assets/icon-192.png',
      vibrate: [100, 50, 100],
      data: { url: data.url || '/' }
    };
    event.waitUntil(
      self.registration.showNotification(data.title || 'InsightAnalyzerAI', options)
    );
  }
});

// 通知クリック
self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  event.waitUntil(
    clients.openWindow(event.notification.data.url)
  );
});
