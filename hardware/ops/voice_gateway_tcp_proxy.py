#!/usr/bin/env python3
"""Proxy TCP traffic from a LAN-visible socket to a loopback-only voice gateway.

Purpose
-------
Keep the Raspberry Pi on the single thh1986 transcript-first voice path even
when the actual websocket gateway is reachable only through a local SSH tunnel
on the development host. This script is transport-only: it does not inspect or
modify websocket payloads, audio, or wakeword decisions.

Usage
-----
Run the proxy in the foreground::

    python3 hardware/ops/voice_gateway_tcp_proxy.py \
        --listen-host 0.0.0.0 \
        --listen-port 8797 \
        --target-host 127.0.0.1 \
        --target-port 18797

Inputs
------
- ``--listen-host`` / ``--listen-port``: Socket that the Pi connects to.
- ``--target-host`` / ``--target-port``: Existing loopback tunnel endpoint that
  already reaches the real thh1986 voice gateway.

Outputs
-------
- Bounded stderr lifecycle logs.
- Exit code 0 on clean shutdown.

Notes
-----
This proxy is intentionally dumb. If the target tunnel disappears, new
connections fail fast instead of silently falling back to another wake path.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
from collections.abc import Sequence

_CHUNK_BYTES = 64 * 1024


async def _pipe(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    close_writer_on_eof: bool,
) -> None:
    """Copy bytes until EOF and only half-close after a real upstream EOF."""

    while True:
        chunk = await reader.read(_CHUNK_BYTES)
        if not chunk:
            if close_writer_on_eof and writer.can_write_eof():
                with contextlib.suppress(Exception):
                    writer.write_eof()
                    await writer.drain()
            return
        writer.write(chunk)
        await writer.drain()


async def _handle_client(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    *,
    target_host: str,
    target_port: int,
) -> None:
    """Bridge one inbound client socket to the configured target socket."""

    peer = client_writer.get_extra_info("peername")
    target_reader: asyncio.StreamReader | None = None
    target_writer: asyncio.StreamWriter | None = None
    try:
        target_reader, target_writer = await asyncio.open_connection(target_host, target_port)
        upstream = asyncio.create_task(
            _pipe(
                client_reader,
                target_writer,
                close_writer_on_eof=True,
            )
        )
        downstream = asyncio.create_task(
            _pipe(
                target_reader,
                client_writer,
                close_writer_on_eof=False,
            )
        )
        results = await asyncio.gather(upstream, downstream, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                raise result
    except Exception as exc:
        print(f"proxy_connection_failed peer={peer!r} error={type(exc).__name__}:{exc}", flush=True)
    finally:
        if target_writer is not None:
            target_writer.close()
            with contextlib.suppress(Exception):
                await target_writer.wait_closed()
        client_writer.close()
        with contextlib.suppress(Exception):
            await client_writer.wait_closed()


async def _run_proxy(args: argparse.Namespace) -> None:
    """Serve the TCP proxy until SIGINT or SIGTERM arrives."""

    stop_event = asyncio.Event()

    def _request_stop() -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(signum, _request_stop)

    server = await asyncio.start_server(
        lambda reader, writer: _handle_client(
            reader,
            writer,
            target_host=args.target_host,
            target_port=args.target_port,
        ),
        host=args.listen_host,
        port=args.listen_port,
    )
    sockets = ", ".join(str(sock.getsockname()) for sock in (server.sockets or []))
    print(
        f"voice_gateway_tcp_proxy_listening listen={sockets} target={(args.target_host, args.target_port)!r}",
        flush=True,
    )
    async with server:
        await stop_event.wait()


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the transport-only proxy."""

    parser = argparse.ArgumentParser(
        description=(
            "Expose a LAN-visible TCP socket that forwards byte-for-byte to a "
            "loopback-only Twinr voice gateway tunnel."
        )
    )
    parser.add_argument("--listen-host", default="0.0.0.0", help="Host/interface to bind. Defaults to 0.0.0.0.")
    parser.add_argument("--listen-port", type=int, default=8797, help="TCP port to bind. Defaults to 8797.")
    parser.add_argument("--target-host", default="127.0.0.1", help="Target host. Defaults to 127.0.0.1.")
    parser.add_argument("--target-port", type=int, default=18797, help="Target TCP port. Defaults to 18797.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI args, run the proxy, and return a conventional exit code."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        asyncio.run(_run_proxy(args))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
