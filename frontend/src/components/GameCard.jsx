// components/GameCard.jsx

import { Tag } from './ui'

const fmt = {
  price: (p, free) => free ? 'Free' : p ? `$${p}` : '—',
  rev:   r => r >= 1000 ? `${(r / 1000).toFixed(0)}k` : (r || '—'),
  score: s => s ? `${(s * 100).toFixed(0)}%` : '',
}

export function GameCard({ game, onClick, showScore = true }) {
  return (
    <div onClick={() => onClick?.(game)}
      style={{
        background: 'var(--surface)',
        border: '1px solid var(--border)',
        borderRadius: 10,
        overflow: 'hidden',
        cursor: 'pointer',
        transition: 'all .2s',
        position: 'relative',
      }}
      onMouseEnter={e => {
        e.currentTarget.style.borderColor = 'var(--border2)'
        e.currentTarget.style.transform = 'translateY(-2px)'
        e.currentTarget.style.boxShadow = '0 8px 24px rgba(0,0,0,.4)'
      }}
      onMouseLeave={e => {
        e.currentTarget.style.borderColor = 'var(--border)'
        e.currentTarget.style.transform = ''
        e.currentTarget.style.boxShadow = ''
      }}
    >
      {/* Image */}
      {game.header_image
        ? <img src={game.header_image} alt={game.name}
            style={{ width: '100%', aspectRatio: '460/215', objectFit: 'cover', display: 'block', background: 'var(--surface2)' }}
            loading="lazy"
            onError={e => { e.target.style.display = 'none' }}
          />
        : <div style={{ width: '100%', aspectRatio: '460/215', background: 'var(--surface2)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text3)', fontSize: '.7rem', fontFamily: 'monospace' }}>
            NO IMAGE
          </div>
      }

      {/* Score badge */}
      {showScore && game.score > 0 && (
        <div style={{
          position: 'absolute', top: '.5rem', right: '.5rem',
          background: 'rgba(6,10,15,.9)', border: '1px solid var(--border2)',
          borderRadius: 4, padding: '.15rem .4rem',
          fontSize: '.68rem', color: 'var(--accent)',
          fontFamily: 'JetBrains Mono, monospace',
        }}>{fmt.score(game.score)}</div>
      )}

      {/* Body */}
      <div style={{ padding: '.6rem .75rem .75rem' }}>
        <div style={{
          fontWeight: 600, fontSize: '.82rem', color: 'var(--white)',
          whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
          marginBottom: '.3rem',
        }} title={game.name}>{game.name}</div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: '.75rem', color: 'var(--accent)', fontWeight: 600 }}>
            {fmt.price(game.price_usd, game.is_free)}
          </span>
          <span style={{ fontSize: '.7rem', color: 'var(--text3)' }}>
            {fmt.rev(game.recommendations)}
          </span>
        </div>

        {game.genres?.length > 0 && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '.25rem', marginTop: '.4rem' }}>
            {game.genres.slice(0, 3).map(g => <Tag key={g}>{g}</Tag>)}
          </div>
        )}
      </div>
    </div>
  )
}

export function GameGrid({ games, onCardClick, showScore }) {
  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fill, minmax(185px, 1fr))',
      gap: '.875rem',
    }}>
      {games.map(g => (
        <GameCard key={g.appid} game={g} onClick={onCardClick} showScore={showScore} />
      ))}
    </div>
  )
}