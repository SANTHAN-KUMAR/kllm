"""Sealed rendezvous messages: encrypt-then-MAC with the standard library only.

The kernel and this PC share `rendezvous_secret` (random, made by `kllm init`, baked into the private kernel the
same way the backend API key is). Everything that crosses the public relay (ntfy) is sealed with it:

  - confidentiality: HMAC-SHA256 in counter mode as the keystream (a PRF-CTR stream cipher), fresh 16-byte nonce
  - integrity/authenticity: HMAC-SHA256 over version || nonce || ciphertext, checked in constant time
  - separate keys for encryption, MAC and topic names, all derived from the secret with HMAC

So the relay operator and anyone who guesses the topic see only random bytes, and a forged or altered message
(for example a fake tunnel URL meant to capture prompts) fails the MAC and is dropped. Replay of an old genuine
message is handled by callers through the `ts` field inside the sealed payload.

Keep this file in sync with the copy in kernel/server_kernel.tmpl.py (tests/test_sealbox.py checks they interoperate).
"""
import base64, hashlib, hmac, json, os, secrets

VERSION = b"v1"


def new_secret():
    return secrets.token_urlsafe(32)


def _k(secret, label):
    return hmac.new(secret.encode("utf-8"), label, hashlib.sha256).digest()


def topics(secret):
    """(status_topic, control_topic): unguessable ntfy topic names derived from the secret."""
    return ("kllm-" + _k(secret, b"kllm-topic-status").hex()[:40],
            "kllm-" + _k(secret, b"kllm-topic-control").hex()[:40])


def _stream(key, nonce, n):
    out, i = bytearray(), 0
    while len(out) < n:
        out += hmac.new(key, nonce + i.to_bytes(4, "big"), hashlib.sha256).digest()
        i += 1
    return bytes(out[:n])


def seal(secret, obj):
    pt = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    nonce = os.urandom(16)
    ct = bytes(a ^ b for a, b in zip(pt, _stream(_k(secret, b"kllm-enc"), nonce, len(pt))))
    tag = hmac.new(_k(secret, b"kllm-mac"), VERSION + nonce + ct, hashlib.sha256).digest()
    return "v1." + base64.urlsafe_b64encode(nonce + ct + tag).decode("ascii").rstrip("=")


def unseal(secret, text):
    """The payload dict, or None for anything not sealed with this secret (forged, altered, foreign, garbage)."""
    try:
        if not isinstance(text, str) or not text.startswith("v1."):
            return None
        raw = text[3:]
        raw = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))
        if len(raw) < 16 + 32:
            return None
        nonce, ct, tag = raw[:16], raw[16:-32], raw[-32:]
        want = hmac.new(_k(secret, b"kllm-mac"), VERSION + nonce + ct, hashlib.sha256).digest()
        if not hmac.compare_digest(tag, want):
            return None
        pt = bytes(a ^ b for a, b in zip(ct, _stream(_k(secret, b"kllm-enc"), nonce, len(ct))))
        obj = json.loads(pt.decode("utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None
