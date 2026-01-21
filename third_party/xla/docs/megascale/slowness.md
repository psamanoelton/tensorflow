# Slowness

Stragglers often manifest as outliers (performs significantly differently from
other hosts) in key Megascale metrics. These key metrics include:

*   Transfer latency: Measures the round trip for a transfer. Outliers are often
    correlated with faulty network hardware.
*   Inbound transfer latency: Measures the one-way latency for a transfer.
    Similar to the preceding metric, outliers here are also correlated with
    faulty network hardware.
*   Compute latency: Measures the time it takes to compute a reduction. Outliers
    are correlated with CPU and memory (e.g., DIMM) issues.

The key to finding a straggler is looking for consistent outliers in time-based
metrics across different workers. These metrics are most useful when grouped by
hosts. Once a host is identified, it can be removed from the general pool of
machines or de-preferred.

**Note on Future Tooling:** Google is actively working on open-sourcing versions
of diagnostic dashboards and network analysis tools to provide a more
streamlined experience for Cloud TPU customers to identify and diagnose
stragglers and network performance issues. These will be available soon.