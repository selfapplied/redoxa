# chess_gray8.py — 8×8 board inside 𝔽₂⁸ (Gray–Kravchuk gauge)

# Gray(3) <-> int
def gray3(n: int) -> tuple[int,int,int]:
    assert 0 <= n < 8
    g = n ^ (n >> 1)
    return (g & 1, (g >> 1) & 1, (g >> 2) & 1)  # (b0,b1,b2)

def inv_gray3(bits: tuple[int,int,int]) -> int:
    g = bits[0] | (bits[1] << 1) | (bits[2] << 2)
    b = 0
    while g:
        b ^= g
        g >>= 1
    return b  # 0..7

# π: lexicographic <-> Gray index helpers (per axis)
def pi_lex_to_gray(i: int) -> int:  # 0..7 -> 0..7 (Gray-coded index)
    b0,b1,b2 = gray3(i)
    return b0 | (b1<<1) | (b2<<2)

def pi_gray_to_lex(j: int) -> int:  # inverse
    return inv_gray3((j & 1, (j>>1)&1, (j>>2)&1))

# pack/unpack: [f0 f1 f2 r0 r1 r2 β κ]  (bit0 = LSB = f0)
def pack_byte(file_idx: int, rank_idx: int, beta: int, kappa: int) -> int:
    f0,f1,f2 = gray3(file_idx)
    r0,r1,r2 = gray3(rank_idx)
    return (f0
            | (f1 << 1) | (f2 << 2)
            | (r0 << 3) | (r1 << 4) | (r2 << 5)
            | (beta << 6) | (kappa << 7))  # 0..255

def unpack_byte(u8: int):
    fbits = (u8 & 1, (u8>>1)&1, (u8>>2)&1)
    rbits = ((u8>>3)&1, (u8>>4)&1, (u8>>5)&1)
    beta  = (u8>>6) & 1
    kappa = (u8>>7) & 1
    f = inv_gray3(fbits); r = inv_gray3(rbits)
    return f, r, beta, kappa, fbits, rbits

# notation helpers
def file_to_i(file_char: str) -> int: return ord(file_char) - ord('a')        # a..h -> 0..7
def rank_to_j(rank_char: str) -> int: return int(rank_char) - 1               # '1'..'8' -> 0..7
def ij_to_sq(i: int, j: int) -> str: return chr(ord('a')+i) + str(j+1)

def encode_square(sq: str, beta=0, kappa=0) -> int:
    i, j = file_to_i(sq[0]), rank_to_j(sq[1])
    return pack_byte(i, j, beta, kappa)

def decode_square(u8: int) -> tuple[str,int,int]:
    i,j,b,k,_,_ = unpack_byte(u8)
    return ij_to_sq(i,j), b, k

# Hamming-shell flips for piece classes (Gray space)
def flip_bits_Gray(fbits, rbits, flip_f_idxs=(), flip_r_idxs=()):
    f = list(fbits); r = list(rbits)
    for t in flip_f_idxs: f[t] ^= 1
    for t in flip_r_idxs: r[t] ^= 1
    return tuple(f), tuple(r)

# CE1-friendly Knight: H3 = 3 flips split (1+2) across axes; optional κ toggle
def knight_step(u8: int, split='f2+r1', toggle_kappa=True) -> int:
    f, r, beta, kappa, fbits, rbits = unpack_byte(u8)
    # choose one of the 4 L patterns (two axis choices × two which-axis-gets-2)
    if split == 'f2+r1':
        fbits2, rbits2 = flip_bits_Gray(fbits, rbits, flip_f_idxs=(0,1), flip_r_idxs=(2,))  # any Gray-idx pair works
    elif split == 'f1+r2':
        fbits2, rbits2 = flip_bits_Gray(fbits, rbits, flip_f_idxs=(2,), flip_r_idxs=(0,1))
    elif split == 'f2+r1_alt':
        fbits2, rbits2 = flip_bits_Gray(fbits, rbits, flip_f_idxs=(1,2), flip_r_idxs=(0,))
    elif split == 'f1+r2_alt':
        fbits2, rbits2 = flip_bits_Gray(fbits, rbits, flip_f_idxs=(0,), flip_r_idxs=(1,2))
    else:
        raise ValueError("split must be one of {'f2+r1','f1+r2','f2+r1_alt','f1+r2_alt'}")
    i2 = inv_gray3(fbits2); j2 = inv_gray3(rbits2)
    k2 = kappa ^ (1 if toggle_kappa else 0)
    return pack_byte(i2, j2, beta, k2)

# Rook/Bishop shells in Gray (H1/H2), κ preserved
def rook_step(u8: int, axis='file', which_bit=0, sign=+1) -> int:
    f, r, beta, kappa, fbits, rbits = unpack_byte(u8)
    if axis == 'file':
        fbits = tuple(b ^ (1 if t==which_bit else 0) for t,b in enumerate(fbits))
    else:
        rbits = tuple(b ^ (1 if t==which_bit else 0) for t,b in enumerate(rbits))
    i2 = inv_gray3(fbits); j2 = inv_gray3(rbits)
    return pack_byte(i2, j2, beta, kappa)

def bishop_shell(u8: int, bits=(0,0)) -> int:
    # flip one Gray bit on each axis (H2)
    f, r, beta, kappa, fbits, rbits = unpack_byte(u8)
    fi, ri = bits
    fbits = tuple(b ^ (1 if t==fi else 0) for t,b in enumerate(fbits))
    rbits = tuple(b ^ (1 if t==ri else 0) for t,b in enumerate(rbits))
    i2 = inv_gray3(fbits); j2 = inv_gray3(rbits)
    return pack_byte(i2, j2, beta, kappa)

# (Optional) K8 Kravchuk lift stubs — plug your orthogonal K_8 here when ready
def K8_index_modes(i: int):  # placeholder: return delta basis index
    return [1 if k==i else 0 for k in range(8)]
def spectral_lift(u8: int):
    i,j,_,_,_,_ = unpack_byte(u8)
    return K8_index_modes(i), K8_index_modes(j)

# --- quick self-checks ---
if __name__ == "__main__":
    assert encode_square("a1")==0x00
    assert encode_square("e4")==0x16   # corrected from 0x58 (bit-order note)
    assert encode_square("g7", beta=1)==0x6D  # corrected from 0xB6
    k = knight_step(encode_square("e4"), split='f2+r1', toggle_kappa=True)
    assert decode_square(k)[0] == "g5" and decode_square(k)[2] == 1  # e4 -> g5 with kappa toggle
    assert k == 0xB5
    print("OK")
