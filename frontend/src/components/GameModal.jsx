// components/GameModal.jsx

import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../api'
import { Stars, Btn, Tag, Spinner, toast } from './ui'
import { GameCard } from './GameCard'

export function GameModal({ game, user, onClose }) {
  const qc = useQueryClient()

  // Similar games
  const { data: simData, isLoading: simLoading } = useQuery({
    queryKey: ['similar', game.appid],
    queryFn: () => api.getSimilar(game.appid, 6),
    enabled: !!game,
  })

  // Game status (rating, fav, wishlist)
  const { data: status, refetch: refetchStatus } = useQuery({
    queryKey: ['gameStatus', game.appid],
    queryFn: () => api.getGameStatus(game.appid),
    enabled: !!user,
    initialData: { rating: null, favorite: false, wishlist: false },
  })

  // Track view
  useEffect(() => {
    if (user) api.trackView(game.appid).catch(() => {})
  }, [game.appid])

  // Mutations
  const rateMut = useMutation({
    mutationFn: async (n) => {
      if (status?.rating === n) {
        await api.deleteRating(game.appid)
        return null
      }
      return api.rateGame(game.appid, n)
    },
    onSuccess: (_, n) => {
      refetchStatus()
      qc.invalidateQueries(['recs'])
      toast(status?.rating === n ? 'Оценка удалена' : `Оценка ${n}★ сохранена`)
    },
    onError: () => toast('Ошибка', 'err'),
  })

  const favMut = useMutation({
    mutationFn: () => status?.favorite ? api.removeFavorite(game.appid) : api.addFavorite(game.appid),
    onSuccess: () => {
      refetchStatus()
      qc.invalidateQueries(['favorites'])
      qc.invalidateQueries(['recs'])
      toast(status?.favorite ? 'Убрано из избранного' : '♥ Добавлено в избранное')
    },
    onError: () => toast('Ошибка', 'err'),
  })

  const wishMut = useMutation({
    mutationFn: () => status?.wishlist ? api.removeWishlist(game.appid) : api.addWishlist(game.appid),
    onSuccess: () => {
      refetchStatus()
      qc.invalidateQueries(['wishlist'])
      qc.invalidateQueries(['recs'])
      toast(status?.wishlist ? 'Убрано из вишлиста' : '★ Добавлено в "Играть позже"')
    },
    onError: () => toast('Ошибка', 'err'),
  })

  const handleInteraction = (fn) => {
    if (!user) { toast('Войдите через Steam', 'err'); return }
    fn()
  }

  const price = game.is_free ? 'Бесплатно' : game.price_usd ? `$${game.price_usd}` : '—'
  const similar = simData?.similar || []

  return (
    <div
      onClick={e => { if (e.target === e.currentTarget) onClose() }}
      style={{
        position: 'fixed', inset: 0,
        background: 'rgba(0,0,0,.8)',
        zIndex: 200,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: '1rem',
        backdropFilter: 'blur(6px)',
      }}
    >
      <div style={{
        background: 'var(--surface)',
        border: '1px solid var(--border2)',
        borderRadius: 14,
        width: '100%', maxWidth: 520,
        maxHeight: '88vh', overflowY: 'auto',
      }}>
        {/* Header */}
        <div style={{
          padding: '1.25rem 1.5rem .875rem',
          display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
          borderBottom: '1px solid var(--border)',
          position: 'sticky', top: 0, background: 'var(--surface)', zIndex: 1,
        }}>
          <div style={{ fontWeight: 700, fontSize: '1rem', color: 'var(--white)', lineHeight: 1.3, flex: 1, marginRight: '1rem' }}>
            {game.name}
          </div>
          <button onClick={onClose} style={{ background: 'none', border: 'none', color: 'var(--text2)', fontSize: '1.3rem', cursor: 'pointer', lineHeight: 1, flexShrink: 0 }}>×</button>
        </div>

        {/* Body */}
        <div style={{ padding: '1.25rem 1.5rem' }}>
          {/* Image */}
          {game.header_image && (
            <img src={game.header_image} alt={game.name}
              style={{ width: '100%', borderRadius: 8, aspectRatio: '460/215', objectFit: 'cover', background: 'var(--surface2)', marginBottom: '1rem' }}
            />
          )}

          {/* Meta */}
          <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap', marginBottom: '1rem', fontSize: '.82rem' }}>
            <span style={{ color: 'var(--accent)', fontWeight: 600, fontSize: '.95rem' }}>{price}</span>
            {game.recommendations > 0 && <span style={{ color: 'var(--text2)' }}>{(game.recommendations / 1000).toFixed(1)}k отзывов</span>}
            {game.metacritic > 0 && <span style={{ color: 'var(--text2)' }}>MC: <b style={{ color: 'var(--white)' }}>{game.metacritic}</b></span>}
            {game.score > 0 && <span style={{ color: 'var(--text2)' }}>Score: <b style={{ color: 'var(--accent)' }}>{(game.score * 100).toFixed(0)}%</b></span>}
          </div>

          {/* Genres */}
          {game.genres?.length > 0 && (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '.3rem', marginBottom: '1rem' }}>
              {game.genres.map(g => <Tag key={g}>{g}</Tag>)}
            </div>
          )}

          {/* Actions */}
          <div style={{ display: 'flex', gap: '.6rem', flexWrap: 'wrap', marginBottom: '1.25rem' }}>
            <Btn as="a" href={game.store_url} target="_blank" rel="noreferrer" size="sm">
              ↗ Открыть в Steam
            </Btn>
            <button onClick={() => handleInteraction(favMut.mutate)}
              style={{
                display: 'inline-flex', alignItems: 'center', gap: '.4rem',
                padding: '.35rem .85rem', borderRadius: 6,
                fontFamily: 'inherit', fontSize: '.78rem', fontWeight: 600,
                cursor: 'pointer', transition: 'all .15s',
                background: status?.favorite ? 'var(--accent-dim)' : 'transparent',
                color: status?.favorite ? 'var(--accent)' : 'var(--text2)',
                border: status?.favorite ? '1px solid rgba(0,212,170,.3)' : '1px solid var(--border2)',
              }}
              disabled={favMut.isPending}
            >
              {status?.favorite ? '♥ В избранном' : '♡ Избранное'}
            </button>
            <button onClick={() => handleInteraction(wishMut.mutate)}
              style={{
                display: 'inline-flex', alignItems: 'center', gap: '.4rem',
                padding: '.35rem .85rem', borderRadius: 6,
                fontFamily: 'inherit', fontSize: '.78rem', fontWeight: 600,
                cursor: 'pointer', transition: 'all .15s',
                background: status?.wishlist ? 'rgba(255,184,0,.08)' : 'transparent',
                color: status?.wishlist ? 'var(--gold)' : 'var(--text2)',
                border: status?.wishlist ? '1px solid rgba(255,184,0,.3)' : '1px solid var(--border2)',
              }}
              disabled={wishMut.isPending}
            >
              {status?.wishlist ? '★ Вишлист' : '☆ Играть позже'}
            </button>
          </div>

          {/* Rating */}
          {user && (
            <div style={{ marginBottom: '1.5rem' }}>
              <div style={{ fontSize: '.72rem', color: 'var(--text2)', marginBottom: '.5rem', textTransform: 'uppercase', letterSpacing: '.06em' }}>
                Ваша оценка {status?.rating ? `— ${status.rating}★ (нажми снова чтобы убрать)` : ''}
              </div>
              <Stars current={status?.rating || 0} onChange={n => rateMut.mutate(n)} />
            </div>
          )}

          {/* Similar */}
          <div style={{ fontSize: '.72rem', color: 'var(--text2)', textTransform: 'uppercase', letterSpacing: '.08em', marginBottom: '.75rem', fontWeight: 600 }}>
            Похожие игры
          </div>
          {simLoading
            ? <div style={{ display: 'flex', justifyContent: 'center', padding: '1rem' }}><Spinner /></div>
            : similar.length > 0
              ? <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '.5rem' }}>
                  {similar.map(g => (
                    <GameCard key={g.appid} game={g} showScore={false}
                      onClick={() => window.open(g.store_url, '_blank')}
                    />
                  ))}
                </div>
              : <p style={{ color: 'var(--text3)', fontSize: '.82rem' }}>Не найдено</p>
          }
        </div>
      </div>
    </div>
  )
}