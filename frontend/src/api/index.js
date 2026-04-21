// api/index.js — все запросы к FastAPI через /api proxy

const BASE = '/api'

async function req(method, path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

export const api = {
  // Auth
  getMe:          () => req('GET', '/me'),
  logout:         () => fetch(`${BASE}/auth/logout`),

  // Recommendations
  getRecs:        (n=40) => req('GET', `/recommendations?top_n=${n}`),
  getRecentRecs:  (n=20) => req('GET', `/recommendations/recent?top_n=${n}`),
  getAnonRecs:    (games, n=24) => req('POST', '/recommendations/anon', { owned_games: games, top_n: n }),

  // Search & similar
  search:         (q, limit=8) => req('GET', `/search?q=${encodeURIComponent(q)}&limit=${limit}`),
  getSimilar:     (appid, n=6) => req('GET', `/similar/${appid}?top_n=${n}`),

  // Game status & interactions
  getGameStatus:  appid => req('GET', `/game/${appid}/status`),
  rateGame:       (appid, rating) => req('POST', `/game/${appid}/rate`, { rating }),
  deleteRating:   appid => req('DELETE', `/game/${appid}/rate`),
  addFavorite:    appid => req('POST', `/game/${appid}/favorite`),
  removeFavorite: appid => req('DELETE', `/game/${appid}/favorite`),
  addWishlist:    appid => req('POST', `/game/${appid}/wishlist`),
  removeWishlist: appid => req('DELETE', `/game/${appid}/wishlist`),
  trackView:      appid => req('POST', `/game/${appid}/view`),

  // Collections
  getFavorites:   () => req('GET', '/my/favorites'),
  getWishlist:    () => req('GET', '/my/wishlist'),
}