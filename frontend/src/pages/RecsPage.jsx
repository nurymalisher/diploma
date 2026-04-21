// pages/RecsPage.jsx — персональные рекомендации

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api'
import { GameGrid } from '../components/GameCard'
import { GameModal } from '../components/GameModal'
import { Loading, Empty, SectionHeader, Tabs, SearchInput } from '../components/ui'

export function RecsPage({ user }) {
  const [tab, setTab] = useState('all')
  const [filter, setFilter] = useState('')
  const [modal, setModal] = useState(null)

  const { data: recsData, isLoading: recsLoading } = useQuery({
    queryKey: ['recs'],
    queryFn: () => api.getRecs(48),
    staleTime: 5 * 60 * 1000,
  })

  const { data: recentData, isLoading: recentLoading } = useQuery({
    queryKey: ['recentRecs'],
    queryFn: () => api.getRecentRecs(24),
    staleTime: 5 * 60 * 1000,
  })

  const recs   = recsData?.recommendations   || []
  const recent = recentData?.recommendations || []

  const filtered = tab === 'all'
    ? recs.filter(g => !filter || g.name.toLowerCase().includes(filter.toLowerCase()))
    : recent

  const loading = tab === 'all' ? recsLoading : recentLoading

  // Profile bar
  const lib = user?.library || {}

  return (
    <div style={{ maxWidth: 1320, margin: '0 auto', padding: '1.5rem' }}>
      {/* Profile bar */}
      <div style={{
        background: 'var(--surface)', border: '1px solid var(--border)',
        borderRadius: 10, padding: '.875rem 1.25rem',
        display: 'flex', alignItems: 'center', gap: '1rem',
        marginBottom: '1.5rem',
      }}>
        {user?.avatar && (
          <img src={user.avatar} alt=""
            style={{ width: 42, height: 42, borderRadius: 8, border: '1.5px solid var(--border2)', objectFit: 'cover', flexShrink: 0 }}
          />
        )}
        <div style={{ flex: 1 }}>
          <div style={{ fontWeight: 600, fontSize: '.95rem', color: 'var(--white)' }}>
            {user?.username || 'Пользователь'}
          </div>
          <div style={{ fontSize: '.75rem', color: 'var(--text2)', fontFamily: 'JetBrains Mono, monospace', marginTop: '.1rem' }}>
            {lib.total_games && <span style={{ marginRight: '1rem' }}>{lib.total_games.toLocaleString()} игр в Steam</span>}
          </div>
        </div>
      </div>

      {/* Stats */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '.75rem', marginBottom: '1.5rem' }}>
        {[
          ['Рекомендаций', recs.length],
          ['Игр в Steam', lib.total_games || '—'],
          ['За 2 недели', recent.length],
        ].map(([lbl, val]) => (
          <div key={lbl} style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8, padding: '.875rem 1rem' }}>
            <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '1.6rem', fontWeight: 500, color: 'var(--accent)', lineHeight: 1 }}>{val}</div>
            <div style={{ fontSize: '.72rem', color: 'var(--text2)', marginTop: '.2rem', textTransform: 'uppercase', letterSpacing: '.05em' }}>{lbl}</div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <Tabs
        tabs={[['all', 'Все рекомендации'], ['recent', 'По недавним играм']]}
        active={tab}
        onChange={setTab}
      />

      {/* Filter */}
      {tab === 'all' && (
        <div style={{ marginBottom: '1.25rem' }}>
          <SearchInput value={filter} onChange={setFilter} placeholder="Фильтр по названию..." />
        </div>
      )}

      {/* Games */}
      {loading
        ? <Loading text="Загружаем рекомендации..." />
        : filtered.length > 0
          ? <>
              <SectionHeader
                title={tab === 'all' ? 'Рекомендации' : 'По недавним играм'}
                count={filtered.length}
              />
              <GameGrid games={filtered} onCardClick={setModal} />
            </>
          : <Empty text={tab === 'recent' ? 'Нет активности за последние 2 недели' : 'Рекомендации не найдены'} />
      }

      {modal && <GameModal game={modal} user={user} onClose={() => setModal(null)} />}
    </div>
  )
}