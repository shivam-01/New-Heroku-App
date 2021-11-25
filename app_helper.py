import base64

def encode_to_base64(byte):
    b64_string = base64.b64encode(byte)
    return b64_string