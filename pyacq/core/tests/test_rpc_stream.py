import numpy as np

from pyacq import ProcessSpawner, OutputStream, RPCServer


def test_steaming_over_rpc():
    proc = ProcessSpawner()
    local_server = RPCServer()
    local_server.run_lazy()
    try:
        assert local_server.running()
        assert RPCServer.get_server() == local_server
        out_stream = OutputStream()
        shape = (100, 100)
        out_stream.configure(shape=shape, dtype='float32')
        in_stream = proc.client._import('pyacq').InputStream()
        in_stream.connect(out_stream)
        assert not in_stream.poll(0)
        ones = np.ones(shape, dtype='float32')
        out_stream.send(ones)
        assert in_stream.poll(0)
        pos, data = in_stream.recv()
        assert np.all(data == ones)
        assert not in_stream.poll(0)
    finally:
        local_server.close()
        proc.stop()
