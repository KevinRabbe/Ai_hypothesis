export function Scaling() {
  return (
    <div className="page-stack">
      <section className="panel">
        <h2>Single-Worker Scaling Context</h2>
        <div className="bucket-row">
          {["10K", "25K", "50K", "75K", "100K", "1M", "10M"].map((label) => (
            <div className="bucket" key={label}>{label}<span>Not available</span></div>
          ))}
        </div>
      </section>
      <section className="panel">
        <h2>Population Organization Matrix</h2>
        <table>
          <thead><tr><th>Total Budget</th><th>25K workers</th><th>50K workers</th><th>75K workers</th></tr></thead>
          <tbody>
            {["~1M total", "~5M total", "~10M total"].map((row) => (
              <tr key={row}><th>{row}</th><td>—</td><td>—</td><td>—</td></tr>
            ))}
          </tbody>
        </table>
      </section>
      <section className="panel">
        <h2>Dense Baseline Context</h2>
        <p>No dense baseline results are currently available.</p>
      </section>
    </div>
  );
}
