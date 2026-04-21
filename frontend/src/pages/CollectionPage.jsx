// pages/CollectionPage.jsx — избранное и вишлист

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api'
import { GameGrid } from '../components/GameCard'
import { GameModal } from '../components/GameModal'
import { Loading, Empty, SectionHeader } from '../components/ui'

export function FavoritesPage({ user }) {
  const [modal, setModal] = useState(null)
  const { data, isLoading } = useQuery({
    queryKey: ['favorites'],
    queryFn: api.getFavorites,
  })
  const games = data?.favorites || []

  return (
    <div style={{ maxWidth: 1320, margin: '0 auto', padding: '1.5rem' }}>
      <SectionHeader title="Избранное" count={games.length} />
      {isLoading
        ? <Loading />
        : games.length > 0
          ? <GameGrid games={games} onCardClick={setModal} showScore={false} />
          : <Empty text="Вы ещё ничего не добавили в избранное. Открой карточку игры и нажми ♡" />
      }
      {modal && <GameModal game={modal} user={user} onClose={() => setModal(null)} />}
    </div>
  )
}

export function WishlistPage({ user }) {
  const [modal, setModal] = useState(null)
  const { data, isLoading } = useQuery({
    queryKey: ['wishlist'],
    queryFn: api.getWishlist,
  })
  const games = data?.wishlist || []

  return (
    <div style={{ maxWidth: 1320, margin: '0 auto', padding: '1.5rem' }}>
      <SectionHeader title="Играть позже" count={games.length} />
      {isLoading
        ? <Loading />
        : games.length > 0
          ? <GameGrid games={games} onCardClick={setModal} showScore={false} />
          : <Empty text="Вы ещё ничего не добавили в вишлист. Открой карточку игры и нажми ☆" />
      }
      {modal && <GameModal game={modal} user={user} onClose={() => setModal(null)} />}
    </div>
  )
}

// Discover page (anon рекомендации для авторизованных)
import { useRef, useEffect } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Btn, Spinner, toast } from '../components/ui'

export function DiscoverPage({ user }) {
  const [q, setQ] = useState('')
  const [results, setResults] = useState([])
  const [picked, setPicked] = useState([])
  const [recs, setRecs] = useState([])
  const [searching, setSearching] = useState(false)
  const [modal, setModal] = useState(null)
  const debRef = useRef(null)
  const dropRef = useRef(null)

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

  useEffect(() => {
    const h = e => { if (!dropRef.current?.contains(e.target)) setResults([]) }
    document.addEventListener('mousedown', h)
    return () => document.removeEventListener('mousedown', h)
  }, [])

  const add = g => {
    if (picked.find(p => p.appid === g.appid)) { toast('Уже добавлено', 'err'); return }
    if (picked.length >= 8) { toast('Максимум 8 игр', 'err'); return }
    setPicked(p => [...p, g]); setQ(''); setResults([])
  }

  const recsMut = useMutation({
    mutationFn: () => api.getAnonRecs(picked.map(g => ({ appid: g.appid, playtime_hours: 50 })), 24),
    onSuccess: d => { setRecs(d.recommendations || []); if (!d.recommendations?.length) toast('Не найдено', 'err') },
    onError: () => toast('Ошибка', 'err'),
  })

  return (
    <div style={{ maxWidth: 1320, margin: '0 auto', padding: '1.5rem' }}>
      <SectionHeader title="Поиск по вкусу" />
      <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 10, padding: '1.25rem', marginBottom: '1.5rem' }}>
        <p style={{ fontSize: '.85rem', color: 'var(--text2)', marginBottom: '1rem' }}>
          Выбери любимые игры — получи рекомендации без учёта Steam библиотеки
        </p>

        <div ref={dropRef} style={{ position: 'relative', marginBottom: '.75rem' }}>
          <svg style={{ position: 'absolute', left: '.875rem', top: '50%', transform: 'translateY(-50%)', color: 'var(--text3)', pointerEvents: 'none', zIndex: 1 }}
            width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>
          </svg>
          <input value={q} onChange={e => setQ(e.target.value)} placeholder="Поиск игры..."
            style={{ width: '100%', background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 8, padding: '.65rem .875rem .65rem 2.5rem', color: 'var(--text)', fontSize: '.88rem', outline: 'none', fontFamily: 'inherit' }}
          />
          {(results.length > 0 || searching) && (
            <div style={{ position: 'absolute', top: '100%', left: 0, right: 0, background: 'var(--surface2)', border: '1px solid var(--border2)', borderTop: 'none', borderRadius: '0 0 8px 8px', zIndex: 50, maxHeight: 280, overflowY: 'auto' }}>
              {searching && <div style={{ padding: '.75rem', color: 'var(--text3)', fontSize: '.8rem', display: 'flex', alignItems: 'center', gap: '.5rem' }}><Spinner size={14} /> Поиск...</div>}
              {results.map(g => (
                <div key={g.appid} onClick={() => add(g)}
                  style={{ display: 'flex', alignItems: 'center', gap: '.75rem', padding: '.55rem .875rem', cursor: 'pointer', borderBottom: '1px solid var(--border)' }}
                  onMouseEnter={e => e.currentTarget.style.background = 'var(--accent-dim)'}
                  onMouseLeave={e => e.currentTarget.style.background = ''}
                >
                  {g.header_image ? <img src={g.header_image} alt="" style={{ width: 72, height: 34, borderRadius: 4, objectFit: 'cover' }} /> : <div style={{ width: 72, height: 34, background: 'var(--border)', borderRadius: 4 }} />}
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: '.83rem', fontWeight: 500, color: 'var(--white)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{g.name}</div>
                  </div>
                  <span style={{ color: 'var(--accent)', fontSize: '.72rem', fontWeight: 600 }}>+ Add</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {picked.length > 0 && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '.4rem', marginBottom: '1rem' }}>
            {picked.map(g => (
              <span key={g.appid} style={{ display: 'inline-flex', alignItems: 'center', gap: '.35rem', background: 'var(--accent-dim)', border: '1px solid rgba(0,212,170,.2)', borderRadius: 20, padding: '.25rem .65rem', fontSize: '.78rem', color: 'var(--accent)' }}>
                {g.name}
                <button onClick={() => setPicked(p => p.filter(x => x.appid !== g.appid))} style={{ background: 'none', border: 'none', color: 'var(--accent)', cursor: 'pointer', opacity: .7 }}>×</button>
              </span>
            ))}
          </div>
        )}

        <Btn onClick={() => recsMut.mutate()} disabled={recsMut.isPending || !picked.length}>
          {recsMut.isPending ? <><Spinner size={14} /> Загрузка...</> : 'Получить рекомендации'}
        </Btn>
      </div>

      {recs.length > 0 && (
        <>
          <SectionHeader title="Результат" count={recs.length} />
          <GameGrid games={recs} onCardClick={setModal} />
        </>
      )}
      {modal && <GameModal game={modal} user={user} onClose={() => setModal(null)} />}
    </div>
  )
}