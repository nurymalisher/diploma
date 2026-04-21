// pages/HomePage.jsx — лендинг для неавторизованных

import { useState, useRef, useEffect } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { api } from '../api'
import { GameGrid } from '../components/GameCard'
import { GameModal } from '../components/GameModal'
import { Btn, Spinner, Empty, SectionHeader, toast } from '../components/ui'

function SteamIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
      <path d="M11.979 0C5.678 0 .511 4.86.022 11.037l6.432 2.658c.545-.371 1.203-.59 1.912-.59.063 0 .125.004.188.006l2.861-4.142V8.91c0-2.495 2.028-4.524 4.524-4.524 2.494 0 4.524 2.031 4.524 4.527s-2.03 4.525-4.524 4.525h-.105l-4.076 2.911c0 .052.004.105.004.159 0 1.875-1.515 3.396-3.39 3.396-1.635 0-3.016-1.173-3.331-2.727L.436 15.27C1.862 20.307 6.486 24 11.979 24c6.627 0 11.999-5.373 11.999-12S18.606 0 11.979 0z"/>
    </svg>
  )
}

export function HomePage() {
  const [q, setQ] = useState('')
  const [results, setResults] = useState([])
  const [picked, setPicked] = useState([])
  const [recs, setRecs] = useState([])
  const [searching, setSearching] = useState(false)
  const [modal, setModal] = useState(null)
  const debRef = useRef(null)
  const dropRef = useRef(null)

  // Search debounce
  useEffect(() => {
    if (q.length < 2) { setResults([]); return }
    clearTimeout(debRef.current)
    debRef.current = setTimeout(() => {
      setSearching(true)
      api.search(q, 8)
        .then(d => { setResults(d.results || []); setSearching(false) })
        .catch(() => setSearching(false))
    }, 300)
  }, [q])

  // Close dropdown on outside click
  useEffect(() => {
    const handler = e => { if (!dropRef.current?.contains(e.target)) setResults([]) }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const addGame = g => {
    if (picked.find(p => p.appid === g.appid)) { toast('Уже добавлено', 'err'); return }
    if (picked.length >= 8) { toast('Максимум 8 игр', 'err'); return }
    setPicked(p => [...p, g])
    setQ(''); setResults([])
  }

  const removeGame = id => setPicked(p => p.filter(x => x.appid !== id))

  const recsMut = useMutation({
    mutationFn: () => api.getAnonRecs(
      picked.map(g => ({ appid: g.appid, playtime_hours: 50 })), 24
    ),
    onSuccess: d => {
      setRecs(d.recommendations || [])
      if (!d.recommendations?.length) toast('Рекомендации не найдены', 'err')
      else toast(`Найдено ${d.recommendations.length} игр`)
    },
    onError: () => toast('Ошибка сервера', 'err'),
  })

  return (
    <div style={{ maxWidth: 1320, margin: '0 auto', padding: '1.5rem' }}>
      {/* Hero */}
      <div style={{ textAlign: 'center', padding: '5rem 0 3.5rem', position: 'relative' }}>
        {/*<div style={{*/}
        {/*  display: 'inline-flex', alignItems: 'center', gap: '.5rem',*/}
        {/*  background: 'var(--accent-dim)', border: '1px solid rgba(0,212,170,.2)',*/}
        {/*  borderRadius: 20, padding: '.3rem .9rem',*/}
        {/*  fontSize: '.75rem', color: 'var(--accent)',*/}
        {/*  fontFamily: 'JetBrains Mono, monospace',*/}
        {/*  marginBottom: '1.75rem',*/}
        {/*}}>*/}
        {/*  <span>●</span> TF-IDF · SVD · Embeddings · Hybrid*/}
        {/*</div>*/}

        <h1 style={{
          fontSize: 'clamp(2.2rem, 5vw, 3.8rem)',
          fontWeight: 700, letterSpacing: '-.04em', lineHeight: 1.1,
          color: 'var(--white)', marginBottom: '1.2rem',
        }}>
          Игры которые<br/>
          <span style={{ color: 'var(--accent)' }}>тебе понравятся</span>
        </h1>

        <p style={{ color: 'var(--text2)', fontSize: '1rem', maxWidth: 460, margin: '0 auto 2.5rem' }}>
          Войди через Steam — система изучит твою библиотеку и подберёт игры специально для тебя с помощью трёх ML-моделей
        </p>

        <a href="/api/auth/steam"
          style={{
            display: 'inline-flex', alignItems: 'center', gap: '.75rem',
            padding: '.85rem 2rem',
            background: 'linear-gradient(135deg,#1b2838,#2a475e)',
            border: '1px solid #4c6b8a',
            borderRadius: 10, color: '#c6d4df',
            fontWeight: 600, fontSize: '.95rem',
            textDecoration: 'none',
            transition: 'all .25s',
          }}
          onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--accent)'; e.currentTarget.style.color = 'var(--white)'; e.currentTarget.style.transform = 'translateY(-2px)'; e.currentTarget.style.boxShadow = '0 8px 30px rgba(0,212,170,.2)' }}
          onMouseLeave={e => { e.currentTarget.style.borderColor = '#4c6b8a'; e.currentTarget.style.color = '#c6d4df'; e.currentTarget.style.transform = ''; e.currentTarget.style.boxShadow = '' }}
        >
          <SteamIcon /> Войти через Steam
        </a>
      </div>

      {/* Divider */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', margin: '0 0 2rem', color: 'var(--text3)', fontSize: '.78rem', fontFamily: 'monospace' }}>
        <div style={{ flex: 1, height: 1, background: 'var(--border)' }} />
        или получи рекомендации без входа
        <div style={{ flex: 1, height: 1, background: 'var(--border)' }} />
      </div>

      {/* Anon panel */}
      <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 10, padding: '1.25rem', marginBottom: '2rem' }}>
        <SectionHeader title="Рекомендации без входа" />
        <p style={{ fontSize: '.85rem', color: 'var(--text2)', marginBottom: '1rem' }}>
          Выбери игры которые тебе нравятся — система подберёт похожие
        </p>

        {/* Search */}
        <div ref={dropRef} style={{ position: 'relative', marginBottom: '.75rem' }}>
          <svg style={{ position: 'absolute', left: '.875rem', top: '50%', transform: 'translateY(-50%)', color: 'var(--text3)', pointerEvents: 'none', zIndex: 1 }}
            width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>
          </svg>
          <input value={q} onChange={e => setQ(e.target.value)}
            placeholder="Поиск игры по названию..."
            style={{
              width: '100%', background: 'var(--surface2)',
              border: '1px solid var(--border)', borderRadius: 8,
              padding: '.65rem .875rem .65rem 2.5rem',
              color: 'var(--text)', fontSize: '.88rem', outline: 'none',
              fontFamily: 'inherit',
            }}
          />

          {/* Dropdown */}
          {(results.length > 0 || searching) && (
            <div style={{
              position: 'absolute', top: '100%', left: 0, right: 0,
              background: 'var(--surface2)', border: '1px solid var(--border2)',
              borderTop: 'none', borderRadius: '0 0 8px 8px',
              zIndex: 50, maxHeight: 320, overflowY: 'auto',
            }}>
              {searching && (
                <div style={{ padding: '.75rem 1rem', color: 'var(--text3)', fontSize: '.8rem', display: 'flex', alignItems: 'center', gap: '.5rem' }}>
                  <Spinner size={14} /> Поиск...
                </div>
              )}
              {results.map(g => (
                <div key={g.appid} onClick={() => addGame(g)}
                  style={{ display: 'flex', alignItems: 'center', gap: '.75rem', padding: '.55rem .875rem', cursor: 'pointer', borderBottom: '1px solid var(--border)', transition: 'background .1s' }}
                  onMouseEnter={e => e.currentTarget.style.background = 'var(--accent-dim)'}
                  onMouseLeave={e => e.currentTarget.style.background = ''}
                >
                  {g.header_image
                    ? <img src={g.header_image} alt="" style={{ width: 72, height: 34, borderRadius: 4, objectFit: 'cover', flexShrink: 0 }} />
                    : <div style={{ width: 72, height: 34, background: 'var(--border)', borderRadius: 4, flexShrink: 0 }} />
                  }
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: '.83rem', fontWeight: 500, color: 'var(--white)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{g.name}</div>
                    <div style={{ fontSize: '.72rem', color: 'var(--text2)' }}>{g.genres?.slice(0, 2).join(', ')}</div>
                  </div>
                  <span style={{ fontSize: '.72rem', color: 'var(--accent)', fontWeight: 600, fontFamily: 'monospace', flexShrink: 0 }}>+ Add</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Picked chips */}
        {picked.length > 0 && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '.4rem', marginBottom: '1rem' }}>
            {picked.map(g => (
              <span key={g.appid} style={{
                display: 'inline-flex', alignItems: 'center', gap: '.35rem',
                background: 'var(--accent-dim)', border: '1px solid rgba(0,212,170,.2)',
                borderRadius: 20, padding: '.25rem .65rem',
                fontSize: '.78rem', color: 'var(--accent)',
              }}>
                {g.name}
                <button onClick={() => removeGame(g.appid)}
                  style={{ background: 'none', border: 'none', color: 'var(--accent)', cursor: 'pointer', fontSize: '.9rem', lineHeight: 1, opacity: .7 }}>×</button>
              </span>
            ))}
          </div>
        )}

        <Btn onClick={() => recsMut.mutate()} disabled={recsMut.isPending || !picked.length}>
          {recsMut.isPending ? <><Spinner size={14} /> Загрузка...</> : 'Получить рекомендации'}
        </Btn>
      </div>

      {/* Anon results */}
      {recs.length > 0 && (
        <>
          <SectionHeader title="Рекомендации для вас" count={recs.length} />
          <GameGrid games={recs} onCardClick={setModal} />
        </>
      )}

      {modal && <GameModal game={modal} user={null} onClose={() => setModal(null)} />}
    </div>
  )
}