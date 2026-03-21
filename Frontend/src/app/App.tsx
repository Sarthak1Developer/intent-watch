import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { LiveFeed } from './pages/LiveFeed';
import { Analytics } from './pages/Analytics';
import { AlertsLog } from './pages/AlertsLog';
import { ZoneConfig } from './pages/ZoneConfig';
import { Settings } from './pages/Settings';
import { History } from './pages/History';

function PageTransition({ children }: { children: React.ReactNode }) {
  const reduceMotion = useReducedMotion();

  if (reduceMotion) {
    return <>{children}</>;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.18, ease: 'easeOut' }}
    >
      {children}
    </motion.div>
  );
}

function AppRoutes() {
  const location = useLocation();

  return (
    <Layout>
      <AnimatePresence mode="wait" initial={false}>
        <Routes location={location} key={location.pathname}>
          <Route path="/" element={<PageTransition><Dashboard /></PageTransition>} />
          <Route path="/feed" element={<PageTransition><LiveFeed /></PageTransition>} />
          <Route path="/analytics" element={<PageTransition><Analytics /></PageTransition>} />
          <Route path="/alerts" element={<PageTransition><AlertsLog /></PageTransition>} />
          <Route path="/history" element={<PageTransition><History /></PageTransition>} />
          <Route path="/zones" element={<PageTransition><ZoneConfig /></PageTransition>} />
          <Route path="/settings" element={<PageTransition><Settings /></PageTransition>} />
        </Routes>
      </AnimatePresence>
    </Layout>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  );
}
