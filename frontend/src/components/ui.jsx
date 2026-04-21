// components/ui.jsx — базовые UI компоненты

import { useState, useEffect, useRef } from 'react'

// ── Toast ─────────────────────────────────────────────────────────────────────
let _toastFn = null
export function setToastFn(fn) { _toastFn = fn }
export function toast(msg, type = 'ok') { _toastFn?.(msg, type) }

export function ToastContainer() {
  const [toasts, setToasts] = useState([])
  useEffect(() => {
    setToastFn((msg, type) => {
      const id = Date.now()
      setToasts(t => [...t, { id, msg, type }])
      setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 3200)
    })
  }, [])
  return (
    <div style={{ position: 'fixed', bottom: '1.25rem', right: '1.25rem', zIndex: 9999, display: 'flex', flexDirection: 'column', gap: '.4rem' }}>
      {toasts.map(t => (
        <div key={t.id} style={{
          background: 'var(--surface2)',
          border: `1px solid var(--border2)`,
          borderLeft: `3px solid ${t.type === 'err' ? 'var(--red)' : 'var(--accent)'}`,
          borderRadius: 'var(--radius)',
          padding: '.6rem 1rem',
          fontSize: '.82rem',
          color: 'var(--text)',
          maxWidth: 300,
          animation: 'slideIn .25s ease',
          boxShadow: '0 4px 20px rgba(0,0,0,.4)',
        }}>{t.msg}</div>
      ))}
      <style>{`@keyframes slideIn{from{opacity:0;transform:translateX(20px)}}`}</style>
    </div>
  )
}

// ── Spinner ───────────────────────────────────────────────────────────────────
export function Spinner({ size = 20 }) {
  return (
    <div style={{
      width: size, height: size,
      border: '2px solid var(--border2)',
      borderTopColor: 'var(--accent)',
      borderRadius: '50%',
      animation: 'spin .65s linear infinite',
      flexShrink: 0,
    }}>
      <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
    </div>
  )
}

// ── Loading ───────────────────────────────────────────────────────────────────
export function Loading({ text = 'Загрузка...' }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '.75rem', padding: '3rem', color: 'var(--text2)' }}>
      <Spinner size={24} /><span>{text}</span>
    </div>
  )
}

// ── Empty ─────────────────────────────────────────────────────────────────────
export function Empty({ text = 'Ничего не найдено' }) {
  return (
    <div style={{ textAlign: 'center', padding: '3rem', color: 'var(--text3)' }}>
      <div style={{ fontSize: '2rem', marginBottom: '.5rem', opacity: .3 }}>◎</div>
      <p style={{ fontSize: '.88rem' }}>{text}</p>
    </div>
  )
}

// ── Btn ───────────────────────────────────────────────────────────────────────
export function Btn({ children, variant = 'accent', size = 'md', as: Tag = 'button', ...props }) {
  const styles = {
    accent: { background: 'var(--accent)', color: '#060A0F', border: 'none' },
    ghost:  { background: 'transparent', color: 'var(--text2)', border: '1px solid var(--border2)' },
    danger: { background: 'transparent', color: 'var(--red)', border: '1px solid rgba(255,71,87,.3)' },
  }
  const sizes = {
    sm: { padding: '.3rem .8rem', fontSize: '.78rem' },
    md: { padding: '.5rem 1.2rem', fontSize: '.85rem' },
    lg: { padding: '.75rem 1.8rem', fontSize: '.95rem' },
  }
  return (
    <Tag style={{
      display: 'inline-flex', alignItems: 'center', gap: '.4rem',
      borderRadius: 7, fontFamily: 'inherit', fontWeight: 600,
      cursor: 'pointer', transition: 'all .2s',
      ...styles[variant], ...sizes[size],
    }} {...props}>{children}</Tag>
  )
}

// ── Tag ───────────────────────────────────────────────────────────────────────
export function Tag({ children }) {
  return (
    <span style={{
      fontSize: '.65rem', padding: '.12rem .4rem',
      background: 'rgba(0,212,170,.08)',
      color: 'var(--accent2)',
      borderRadius: 3, fontWeight: 500,
    }}>{children}</span>
  )
}

// ── Stars ─────────────────────────────────────────────────────────────────────
export function Stars({ current, onChange }) {
  const [hover, setHover] = useState(0)
  return (
    <div style={{ display: 'flex', gap: '.25rem' }}>
      {[1,2,3,4,5].map(n => (
        <button key={n}
          onMouseEnter={() => setHover(n)}
          onMouseLeave={() => setHover(0)}
          onClick={() => onChange?.(n)}
          style={{
            background: 'none', border: 'none', cursor: 'pointer',
            fontSize: '1.1rem', lineHeight: 1,
            color: (hover || current) >= n ? 'var(--gold)' : 'var(--border2)',
            transform: (hover || current) >= n ? 'scale(1.15)' : 'scale(1)',
            transition: 'all .1s',
          }}>★</button>
      ))}
    </div>
  )
}

// ── SearchInput ───────────────────────────────────────────────────────────────
export function SearchInput({ value, onChange, placeholder, children }) {
  return (
    <div style={{ position: 'relative' }}>
      <svg style={{ position: 'absolute', left: '.875rem', top: '50%', transform: 'translateY(-50%)', color: 'var(--text3)', pointerEvents: 'none' }}
        width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>
      </svg>
      <input value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder}
        style={{
          width: '100%', background: 'var(--surface)', border: '1px solid var(--border)',
          borderRadius: 'var(--radius)', padding: '.65rem .875rem .65rem 2.5rem',
          color: 'var(--text)', fontSize: '.88rem', outline: 'none',
          transition: 'border-color .15s', fontFamily: 'inherit',
        }}
        onFocus={e => e.target.style.borderColor = 'var(--accent2)'}
        onBlur={e => e.target.style.borderColor = 'var(--border)'}
      />
      {children}
    </div>
  )
}

// ── SectionHeader ─────────────────────────────────────────────────────────────
export function SectionHeader({ title, count }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '.75rem', marginBottom: '1.25rem' }}>
      <div style={{ fontWeight: 600, fontSize: '1rem', color: 'var(--white)' }}>{title}</div>
      {count != null && (
        <div style={{
          fontFamily: 'JetBrains Mono, monospace', fontSize: '.72rem',
          color: 'var(--text3)', background: 'var(--surface2)',
          padding: '.15rem .45rem', borderRadius: 4,
        }}>{count}</div>
      )}
      <div style={{ flex: 1, height: 1, background: 'var(--border)' }} />
    </div>
  )
}

// ── Tabs ──────────────────────────────────────────────────────────────────────
export function Tabs({ tabs, active, onChange }) {
  return (
    <div style={{ display: 'flex', gap: '.25rem', borderBottom: '1px solid var(--border)', marginBottom: '1.25rem' }}>
      {tabs.map(([k, l]) => (
        <button key={k} onClick={() => onChange(k)}
          style={{
            padding: '.5rem 1rem', borderRadius: '6px 6px 0 0',
            cursor: 'pointer', fontSize: '.83rem', fontFamily: 'inherit',
            background: active === k ? 'var(--surface)' : 'transparent',
            color: active === k ? 'var(--accent)' : 'var(--text2)',
            border: active === k ? '1px solid var(--border)' : '1px solid transparent',
            borderBottom: active === k ? '1px solid var(--surface)' : '1px solid transparent',
            position: 'relative', bottom: -1,
            transition: 'all .15s',
          }}>{l}</button>
      ))}
    </div>
  )
}