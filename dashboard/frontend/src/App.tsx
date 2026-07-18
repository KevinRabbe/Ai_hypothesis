import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./layout/AppShell";
import { Compare } from "./pages/Compare";
import { Evidence } from "./pages/Evidence";
import { Experiments } from "./pages/Experiments";
import { Overview } from "./pages/Overview";
import { Population } from "./pages/Population";
import { RunDetail } from "./pages/RunDetail";
import { Scaling } from "./pages/Scaling";

export default function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<Overview />} />
        <Route path="/experiments" element={<Experiments />} />
        <Route path="/experiments/:experimentId" element={<RunDetail />} />
        <Route path="/compare" element={<Compare />} />
        <Route path="/scaling" element={<Scaling />} />
        <Route path="/population" element={<Population />} />
        <Route path="/population/:experimentId" element={<Population />} />
        <Route path="/evidence" element={<Evidence />} />
        <Route path="/evidence/:experimentId" element={<Evidence />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AppShell>
  );
}
