"""Regression coverage for the transport-only voice gateway TCP proxy."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import unittest


_PROXY_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "hardware" / "ops" / "voice_gateway_tcp_proxy.py"
_PROXY_SPEC = importlib.util.spec_from_file_location("voice_gateway_tcp_proxy", _PROXY_SCRIPT_PATH)
assert _PROXY_SPEC is not None and _PROXY_SPEC.loader is not None
_PROXY_MODULE = importlib.util.module_from_spec(_PROXY_SPEC)
_PROXY_SPEC.loader.exec_module(_PROXY_MODULE)


class VoiceGatewayTcpProxyTests(unittest.IsolatedAsyncioTestCase):
    async def test_proxy_keeps_downstream_reply_after_client_finishes_writing(self) -> None:
        received_by_target = bytearray()

        async def _target_handler(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            try:
                while True:
                    chunk = await reader.read(1024)
                    if not chunk:
                        break
                    received_by_target.extend(chunk)
                await asyncio.sleep(0.05)
                writer.write(b"wake_confirmed")
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        target_server = await asyncio.start_server(_target_handler, host="127.0.0.1", port=0)
        target_port = target_server.sockets[0].getsockname()[1]

        proxy_server = await asyncio.start_server(
            lambda reader, writer: _PROXY_MODULE._handle_client(
                reader,
                writer,
                target_host="127.0.0.1",
                target_port=target_port,
            ),
            host="127.0.0.1",
            port=0,
        )
        proxy_port = proxy_server.sockets[0].getsockname()[1]

        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
            writer.write(b"voice_hello")
            await writer.drain()
            if writer.can_write_eof():
                writer.write_eof()
            response = await asyncio.wait_for(reader.read(1024), timeout=2.0)
        finally:
            writer.close()
            await writer.wait_closed()
            proxy_server.close()
            target_server.close()
            await proxy_server.wait_closed()
            await target_server.wait_closed()

        self.assertEqual(bytes(received_by_target), b"voice_hello")
        self.assertEqual(response, b"wake_confirmed")
