import numpy as np

from pyacq import ProcessSpawner, OutputStream, RPCServer


def test_plaindata_streaming_over_rpc():
    _run_rpc_stream_test(transfermode='plaindata')


def test_sharedmem_steaming_over_rpc():
    _run_rpc_stream_test(transfermode='sharedmem', buffer_size=2)


def _run_rpc_stream_test(**kwds):
    proc = ProcessSpawner()
    local_server = RPCServer()
    local_server.run_lazy()
    try:
        assert local_server.running()
        assert RPCServer.get_server() == local_server
        out_stream = OutputStream()
        shape = (100, 100)
        out_stream.configure(shape=shape, dtype='float32', **kwds)
        in_stream = proc.client._import('pyacq').InputStream()
        in_stream.connect(out_stream)
        assert not in_stream.poll(0)
        send_data = np.random.normal(size=shape).astype('float32')
        out_stream.send(send_data)
        assert in_stream.poll(0)
        pos, recv_data = in_stream.recv()
        assert np.all(recv_data == send_data)
        assert not in_stream.poll(0)
    finally:
        local_server.close()
        proc.stop()
