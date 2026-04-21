// App.jsx — главный роутер

import { useQuery } from '@tanstack/react-query'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { api } from './api'
import { Header } from './components/Header'
import { ToastContainer } from './components/ui.jsx'
import { HomePage } from './pages/HomePage'
import { RecsPage } from './pages/RecsPage'
import { FavoritesPage, WishlistPage, DiscoverPage } from './pages/CollectionPage'

function AppInner() {
  const { data: user, isLoading } = useQuery({
    queryKey: ['me'],
    queryFn: api.getMe,
    retry: false,
    staleTime: 5 * 60 * 1000,
  })

  const isAuth = !!user?.steamid

  if (isLoading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh' }}>
        <div style={{ width: 36, height: 36, border: '3px solid var(--border2)', borderTopColor: 'var(--accent)', borderRadius: '50%', animation: 'spin .65s linear infinite' }} />
        <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
      </div>
    )
  }

  return (
    <>
      <Header user={isAuth ? user : null} />
      <Routes>
        {/* Главная — всегда доступна */}
        <Route path="/" element={
          isAuth ? <Navigate to="/recs" replace /> : <HomePage />
        } />

        {/* Авторизованные */}
        <Route path="/recs" element={
          isAuth ? <RecsPage user={user} /> : <Navigate to="/" replace />
        } />
        <Route path="/favorites" element={
          isAuth ? <FavoritesPage user={user} /> : <Navigate to="/" replace />
        } />
        <Route path="/wishlist" element={
          isAuth ? <WishlistPage user={user} /> : <Navigate to="/" replace />
        } />
        <Route path="/discover" element={
          isAuth ? <DiscoverPage user={user} /> : <Navigate to="/" replace />
        } />

        {/* 404 */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      <ToastContainer />
    </>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AppInner />
    </BrowserRouter>
  )
}