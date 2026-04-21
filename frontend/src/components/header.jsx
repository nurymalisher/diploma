// // components/Header.jsx
//
// import { Link, useLocation } from 'react-router-dom'
//
// export function Header({ user }) {
//   const loc = useLocation()
//
//   const navLinks = user
//     ? [
//         ['/recs', 'Рекомендации'],
//         ['/discover', 'Без входа'],
//         ['/favorites', 'Избранное'],
//         ['/wishlist', 'Вишлист'],
//       ]
//     : []
//
//   const logout = () => {
//     fetch('/api/auth/logout').then(() => window.location.href = '/')
//   }
//
//   return (
//     <header style={{
//       position: 'sticky', top: 0, zIndex: 100,
//       background: 'rgba(6,10,15,.92)',
//       backdropFilter: 'blur(20px)',
//       borderBottom: '1px solid var(--border)',
//     }}>
//       <div style={{
//         maxWidth: 1320, margin: '0 auto',
//         padding: '0 1.5rem',
//         height: 56,
//         display: 'flex', alignItems: 'center', gap: '2rem',
//       }}>
//         {/* Logo */}
//         <Link to="/" style={{ display: 'flex', alignItems: 'center', gap: '.5rem', textDecoration: 'none' }}>
//           <div style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--accent)', boxShadow: '0 0 8px var(--accent)' }} />
//           <span style={{ fontFamily: 'JetBrains Mono, monospace', fontWeight: 500, fontSize: '1rem', color: 'var(--white)' }}>
//             GameRec
//           </span>
//         </Link>
//
//         {/* Nav */}
//         <nav style={{ display: 'flex', gap: '.25rem', flex: 1 }}>
//           {navLinks.map(([to, label]) => (
//             <Link key={to} to={to}
//               style={{
//                 padding: '.35rem .85rem',
//                 borderRadius: 6,
//                 fontSize: '.83rem', fontWeight: 500,
//                 color: loc.pathname === to ? 'var(--accent)' : 'var(--text2)',
//                 background: loc.pathname === to ? 'var(--accent-dim)' : 'transparent',
//                 transition: 'all .15s',
//                 textDecoration: 'none',
//               }}
//               onMouseEnter={e => { if (loc.pathname !== to) e.target.style.color = 'var(--text)' }}
//               onMouseLeave={e => { if (loc.pathname !== to) e.target.style.color = 'var(--text2)' }}
//             >{label}</Link>
//           ))}
//         </nav>
//
//         {/* Right */}
//         <div style={{ display: 'flex', alignItems: 'center', gap: '.75rem', marginLeft: 'auto' }}>
//           {user ? (
//             <>
//               {user.avatar && (
//                 <img src={user.avatar} alt=""
//                   style={{ width: 30, height: 30, borderRadius: 6, border: '1px solid var(--border2)', objectFit: 'cover' }}
//                 />
//               )}
//               <span style={{ fontSize: '.83rem', color: 'var(--text2)' }}>{user.username}</span>
//               <button onClick={logout}
//                 style={{
//                   padding: '.3rem .8rem', borderRadius: 6,
//                   background: 'transparent', border: '1px solid rgba(255,71,87,.3)',
//                   color: 'var(--red)', fontSize: '.78rem', fontWeight: 600,
//                   cursor: 'pointer', fontFamily: 'inherit',
//                 }}>Выйти</button>
//             </>
//           ) : (
//             <a href="/api/auth/steam"
//               style={{
//                 padding: '.4rem 1.1rem', borderRadius: 7,
//                 background: 'var(--accent)', color: '#060A0F',
//                 fontSize: '.85rem', fontWeight: 600,
//                 textDecoration: 'none',
//               }}>Войти</a>
//           )}
//         </div>
//       </div>
//     </header>
//   )
// }
// components/Header.jsx

import { useState, useRef, useEffect } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { api } from '../api'
import { GameModal } from './GameModal'
import { Spinner } from './ui'

export function Header({ user }) {
  const loc = useLocation()

  // Стейты для глобального поиска
  const [q, setQ] = useState('')
  const [results, setResults] = useState([])
  const [searching, setSearching] = useState(false)
  const [modalGame, setModalGame] = useState(null)

  const debRef = useRef(null)
  const dropRef = useRef(null)

  const navLinks = user
    ? [
        ['/recs', 'Рекомендации'],
        ['/discover', 'Без входа'],
        ['/favorites', 'Избранное'],
        ['/wishlist', 'Вишлист'],
      ]
    : []

  const logout = () => {
    fetch('/api/auth/logout').then(() => window.location.href = '/')
  }

  // Логика живого поиска
  useEffect(() => {
    if (q.length < 2) { setResults([]); return }
    clearTimeout(debRef.current)
    debRef.current = setTimeout(() => {
      setSearching(true)
      api.search(q, 6)
        .then(d => { setResults(d.results || []); setSearching(false) })
        .catch(() => setSearching(false))
    }, 300)
  }, [q])

  // Закрытие дропдауна при клике вне него
  useEffect(() => {
    const handler = e => { if (!dropRef.current?.contains(e.target)) setResults([]) }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  return (
    <>
      <header style={{
        position: 'sticky', top: 0, zIndex: 100,
        background: 'rgba(6,10,15,.92)',
        backdropFilter: 'blur(20px)',
        borderBottom: '1px solid var(--border)',
      }}>
        <div style={{
          maxWidth: 1320, margin: '0 auto',
          padding: '0 1.5rem',
          height: 56,
          display: 'flex', alignItems: 'center', gap: '2rem',
        }}>
          {/* Logo */}
          <Link to="/" style={{ display: 'flex', alignItems: 'center', gap: '.5rem', textDecoration: 'none', flexShrink: 0 }}>
            <div style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--accent)', boxShadow: '0 0 8px var(--accent)' }} />
            <span style={{ fontFamily: 'JetBrains Mono, monospace', fontWeight: 500, fontSize: '1rem', color: 'var(--white)' }}>
              GameRec
            </span>
          </Link>

          {/* Nav */}
          <nav style={{ display: 'flex', gap: '.25rem', flexShrink: 0 }}>
            {navLinks.map(([to, label]) => (
              <Link key={to} to={to}
                style={{
                  padding: '.35rem .85rem',
                  borderRadius: 6,
                  fontSize: '.83rem', fontWeight: 500,
                  color: loc.pathname === to ? 'var(--accent)' : 'var(--text2)',
                  background: loc.pathname === to ? 'var(--accent-dim)' : 'transparent',
                  transition: 'all .15s',
                  textDecoration: 'none',
                }}
                onMouseEnter={e => { if (loc.pathname !== to) e.target.style.color = 'var(--text)' }}
                onMouseLeave={e => { if (loc.pathname !== to) e.target.style.color = 'var(--text2)' }}
              >{label}</Link>
            ))}
          </nav>

          {/* Глобальный поиск (показываем только авторизованным) */}
          {user && (
            <div ref={dropRef} style={{ position: 'relative', flex: 1, maxWidth: 320, marginLeft: 'auto' }}>
              <svg style={{ position: 'absolute', left: '.75rem', top: '50%', transform: 'translateY(-50%)', color: 'var(--text3)', pointerEvents: 'none', zIndex: 1 }}
                width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>
              </svg>
              <input
                value={q}
                onChange={e => setQ(e.target.value)}
                placeholder="Найти игру для оценки..."
                style={{
                  width: '100%', background: 'var(--surface2)', border: '1px solid var(--border)',
                  borderRadius: 20, padding: '.4rem .875rem .4rem 2.2rem',
                  color: 'var(--text)', fontSize: '.82rem', outline: 'none', fontFamily: 'inherit',
                  transition: 'border-color .2s'
                }}
                onFocus={e => e.target.style.borderColor = 'var(--accent2)'}
                onBlur={e => e.target.style.borderColor = 'var(--border)'}
              />

              {/* Выпадающий список результатов */}
              {(results.length > 0 || searching) && (
                <div style={{
                  position: 'absolute', top: 'calc(100% + 8px)', left: 0, right: 0,
                  background: 'var(--surface2)', border: '1px solid var(--border2)',
                  borderRadius: 8, zIndex: 50, maxHeight: 320, overflowY: 'auto',
                  boxShadow: '0 10px 30px rgba(0,0,0,0.5)'
                }}>
                  {searching && (
                    <div style={{ padding: '.75rem 1rem', color: 'var(--text3)', fontSize: '.8rem', display: 'flex', alignItems: 'center', gap: '.5rem' }}>
                      <Spinner size={14} /> Ищем...
                    </div>
                  )}
                  {results.map(g => (
                    <div key={g.appid}
                      onClick={() => { setModalGame(g); setQ(''); setResults([]); }}
                      style={{ display: 'flex', alignItems: 'center', gap: '.75rem', padding: '.5rem .75rem', cursor: 'pointer', borderBottom: '1px solid var(--border)', transition: 'background .1s' }}
                      onMouseEnter={e => e.currentTarget.style.background = 'var(--surface)'}
                      onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                    >
                      {g.header_image
                        ? <img src={g.header_image} alt="" style={{ width: 46, height: 22, borderRadius: 3, objectFit: 'cover', flexShrink: 0 }} />
                        : <div style={{ width: 46, height: 22, background: 'var(--border)', borderRadius: 3, flexShrink: 0 }} />
                      }
                      <div style={{ fontSize: '.8rem', fontWeight: 500, color: 'var(--white)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {g.name}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Right: User Profile / Login */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '.75rem', marginLeft: user ? '1rem' : 'auto' }}>
            {user ? (
              <>
                {user.avatar && (
                  <img src={user.avatar} alt=""
                    style={{ width: 30, height: 30, borderRadius: 6, border: '1px solid var(--border2)', objectFit: 'cover' }}
                  />
                )}
                <span style={{ fontSize: '.83rem', color: 'var(--text2)' }}>{user.username}</span>
                <button onClick={logout}
                  style={{
                    padding: '.3rem .8rem', borderRadius: 6,
                    background: 'transparent', border: '1px solid rgba(255,71,87,.3)',
                    color: 'var(--red)', fontSize: '.78rem', fontWeight: 600,
                    cursor: 'pointer', fontFamily: 'inherit',
                  }}>Выйти</button>
              </>
            ) : (
              <a href="/api/auth/steam"
                style={{
                  padding: '.4rem 1.1rem', borderRadius: 7,
                  background: 'var(--accent)', color: '#060A0F',
                  fontSize: '.85rem', fontWeight: 600,
                  textDecoration: 'none',
                }}>Войти</a>
            )}
          </div>
        </div>
      </header>

      {/* Если игра выбрана, показываем модалку поверх всего приложения */}
      {modalGame && (
        <GameModal game={modalGame} user={user} onClose={() => setModalGame(null)} />
      )}
    </>
  )
}