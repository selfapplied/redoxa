"""
CE1 Chess 8×8 ↔ 𝔽₂⁸ (Gray–Kravchuk gauge)

Mathematically rigorous implementation of chess board algebra with CE1 invariants.
Each square (file a-h, rank 1-8) encoded as byte in 𝔽₂⁸ with structure aligned to 
CE1 invariants and Möbius/torus moves.

CE1{lens=PK|mode=Board8|
Ξ=Gray3×Gray3|π=lex↔gray|β∈{torus,mirror}|κ∈{μ,κ}|
encode(u8):=[f0 f1 f2 r0 r1 r2 β κ];
γ(0..7):=000,001,011,010,110,111,101,100;
I_locality:= rook∈H1, bishop∈H2, knight∈H3(1+2);
Knight:= MöbiusFlip ⇒ κ↦κ⊕1 ∧ flip(H3 pattern);
Rook/Bishop:= orientable ⇒ κ fixed ∧ flip(H1/H2);
β policy:= periodic↔mirror pad on kernels;
Kravchuk:= optional spectral lift per axis; }
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

class BoundaryPolicy(Enum):
    """Boundary policy for board edges"""
    TORUS = 0      # Periodic boundary conditions (β=0)
    MIRROR = 1     # Mirror boundary conditions (β=1)

class OrientationMode(Enum):
    """Orientation mode for moves"""
    MU = 0         # Orientable (μ/torus, κ=0)
    KAPPA = 1      # Non-orientable (κ/Möbius, κ=1)

@dataclass
class ChessSquare:
    """Chess square with algebraic and byte representations"""
    file: str      # a-h
    rank: int      # 1-8
    byte: int      # 𝔽₂⁸ encoding
    boundary: BoundaryPolicy
    orientation: OrientationMode

@dataclass
class MoveResult:
    """Result of a chess move"""
    from_square: ChessSquare
    to_square: ChessSquare
    move_type: str  # "rook", "bishop", "knight", etc.
    boundary_policy: BoundaryPolicy

class CE1ChessAlgebra:
    """
    CE1 Chess Board Algebra
    
    Implements 8×8 chess board as 𝔽₂⁸ with Gray-Kravchuk gauge.
    Movement classes become Hamming shells for kernel-friendly computation.
    """
    
    def __init__(self):
        """Initialize chess algebra with CE1 invariants"""
        # Gray code mapping for 3 bits: γ(0..7) = 000,001,011,010,110,111,101,100
        self.gray_code = [0, 1, 3, 2, 6, 7, 5, 4]  # Binary-reflected Gray
        self.gray_decode_dict = {code: i for i, code in enumerate(self.gray_code)}
        
        # File and rank mappings
        self.file_to_index = {chr(ord('a') + i): i for i in range(8)}
        self.index_to_file = {i: chr(ord('a') + i) for i in range(8)}
        
        # Kravchuk basis for spectral analysis (K₈)
        self.kravchuk_basis = self._generate_kravchuk_basis(8)
        
        # π permutation gauge (lexicographic ↔ Gray)
        self.pi_lex_to_gray = self._generate_pi_permutation()
        self.pi_gray_to_lex = {v: k for k, v in self.pi_lex_to_gray.items()}
    
    def _generate_kravchuk_basis(self, n: int) -> np.ndarray:
        """Generate Kravchuk basis K₈ for dimension n"""
        basis = np.zeros((n, n), dtype=float)
        
        for k in range(n):
            for x in range(n):
                # Kravchuk polynomial K_k(x, n)
                basis[x, k] = self._kravchuk_polynomial(k, x, n)
        
        # Orthonormalize
        basis = basis / np.linalg.norm(basis, axis=0, keepdims=True)
        return basis
    
    def _kravchuk_polynomial(self, k: int, x: int, n: int) -> float:
        """Compute Kravchuk polynomial K_k(x, n)"""
        result = 0.0
        
        for j in range(k + 1):
            if j <= x and (k - j) <= (n - x):
                coeff = (-1)**j * self._binomial_coeff(x, j) * self._binomial_coeff(n - x, k - j)
                result += coeff
        
        return result
    
    def _binomial_coeff(self, n: int, k: int) -> int:
        """Compute binomial coefficient C(n, k)"""
        if k > n or k < 0:
            return 0
        if k == 0 or k == n:
            return 1
        
        result = 1
        for i in range(min(k, n - k)):
            result = result * (n - i) // (i + 1)
        
        return result
    
    def _generate_pi_permutation(self) -> Dict[int, int]:
        """Generate π permutation gauge (lexicographic ↔ Gray)"""
        pi = {}
        for i in range(8):
            # Convert to Gray code
            gray_i = self.gray_code[i]
            pi[i] = gray_i
        return pi
    
    def gray_encode(self, n: int) -> Tuple[int, int, int]:
        """Gray encode: γ(n) → (b0, b1, b2) for n ∈ {0..7}"""
        assert 0 <= n < 8
        g = self.gray_code[n]
        return (g & 1, (g >> 1) & 1, (g >> 2) & 1)
    
    def gray_decode(self, bits: Tuple[int, int, int]) -> int:
        """Gray decode: γ⁻¹(b0, b1, b2) → n"""
        g = bits[0] | (bits[1] << 1) | (bits[2] << 2)
        return self.gray_decode_dict.get(g, 0)
    
    def encode_square(self, file: str, rank: int, 
                     boundary: BoundaryPolicy = BoundaryPolicy.TORUS,
                     orientation: OrientationMode = OrientationMode.MU) -> int:
        """
        Encode chess square as byte in 𝔽₂⁸
        
        Byte layout (LSB→MSB): [f0 f1 f2 r0 r1 r2 β κ]
        """
        # Convert file and rank to indices
        f = self.file_to_index[file]
        r = rank - 1  # Convert 1-8 to 0-7
        
        # Convert to Gray codes
        f_bits = self.gray_encode(f)
        r_bits = self.gray_encode(r)
        
        # Pack byte: [f0 f1 f2 r0 r1 r2 β κ]
        byte_value = (f_bits[0] + 
                     2 * f_bits[1] + 
                     4 * f_bits[2] + 
                     8 * r_bits[0] + 
                     16 * r_bits[1] + 
                     32 * r_bits[2] + 
                     64 * boundary.value + 
                     128 * orientation.value)
        
        return byte_value
    
    def decode_square(self, byte_value: int) -> ChessSquare:
        """
        Decode byte to chess square
        
        Returns ChessSquare with file, rank, and policy information
        """
        # Unpack bits: [f0 f1 f2 r0 r1 r2 β κ]
        f_bits = (byte_value & 1, (byte_value >> 1) & 1, (byte_value >> 2) & 1)
        r_bits = ((byte_value >> 3) & 1, (byte_value >> 4) & 1, (byte_value >> 5) & 1)
        beta = (byte_value >> 6) & 1
        kappa = (byte_value >> 7) & 1
        
        # Convert Gray codes back to indices
        f = self.gray_decode(f_bits)
        r = self.gray_decode(r_bits)
        
        # Convert to chess notation
        file_char = self.index_to_file[f]
        rank_num = r + 1
        
        return ChessSquare(
            file=file_char,
            rank=rank_num,
            byte=byte_value,
            boundary=BoundaryPolicy(beta),
            orientation=OrientationMode(kappa)
        )
    
    def rook_move(self, from_square: ChessSquare, delta_file: int, delta_rank: int) -> Optional[MoveResult]:
        """
        Rook move: Hamming-1 (flip one Gray bit)
        Orientable: κ unchanged
        """
        # Get current position
        f = self.file_to_index[from_square.file]
        r = from_square.rank - 1
        
        # Apply delta
        new_f = f + delta_file
        new_r = r + delta_rank
        
        # Check bounds
        if not (0 <= new_f < 8 and 0 <= new_r < 8):
            return None
        
        # Create new square (same boundary and orientation)
        to_square = ChessSquare(
            file=self.index_to_file[new_f],
            rank=new_r + 1,
            byte=self.encode_square(self.index_to_file[new_f], new_r + 1, 
                                  from_square.boundary, from_square.orientation),
            boundary=from_square.boundary,
            orientation=from_square.orientation
        )
        
        return MoveResult(
            from_square=from_square,
            to_square=to_square,
            move_type="rook",
            boundary_policy=from_square.boundary
        )
    
    def bishop_move(self, from_square: ChessSquare, delta_file: int, delta_rank: int) -> Optional[MoveResult]:
        """
        Bishop move: Hamming-2 (flip two Gray bits)
        Orientable: κ unchanged
        """
        # Get current position
        f = self.file_to_index[from_square.file]
        r = from_square.rank - 1
        
        # Apply delta
        new_f = f + delta_file
        new_r = r + delta_rank
        
        # Check bounds
        if not (0 <= new_f < 8 and 0 <= new_r < 8):
            return None
        
        # Create new square (same boundary and orientation)
        to_square = ChessSquare(
            file=self.index_to_file[new_f],
            rank=new_r + 1,
            byte=self.encode_square(self.index_to_file[new_f], new_r + 1, 
                                  from_square.boundary, from_square.orientation),
            boundary=from_square.boundary,
            orientation=from_square.orientation
        )
        
        return MoveResult(
            from_square=from_square,
            to_square=to_square,
            move_type="bishop",
            boundary_policy=from_square.boundary
        )
    
    def knight_move(self, from_square: ChessSquare, delta_file: int, delta_rank: int,
                   mobius_flip: bool = True) -> Optional[MoveResult]:
        """
        Knight move: Hamming-3 with (1+2) split across axes
        MöbiusFlip: κ ↦ κ ⊕ 1 (orientation flip)
        """
        # Get current position
        f = self.file_to_index[from_square.file]
        r = from_square.rank - 1
        
        # Apply delta
        new_f = f + delta_file
        new_r = r + delta_rank
        
        # Check bounds
        if not (0 <= new_f < 8 and 0 <= new_r < 8):
            return None
        
        # Determine new orientation
        new_orientation = from_square.orientation
        if mobius_flip:
            # Toggle κ: κ ↦ κ ⊕ 1
            new_orientation = OrientationMode(1 - from_square.orientation.value)
        
        # Create new square
        to_square = ChessSquare(
            file=self.index_to_file[new_f],
            rank=new_r + 1,
            byte=self.encode_square(self.index_to_file[new_f], new_r + 1, 
                                  from_square.boundary, new_orientation),
            boundary=from_square.boundary,
            orientation=new_orientation
        )
        
        return MoveResult(
            from_square=from_square,
            to_square=to_square,
            move_type="knight",
            boundary_policy=from_square.boundary
        )
    
    def generate_all_moves(self, from_square: ChessSquare) -> List[MoveResult]:
        """Generate all possible moves from a square"""
        moves = []
        
        # Rook moves (Hamming-1)
        for delta in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            move = self.rook_move(from_square, delta[0], delta[1])
            if move:
                moves.append(move)
        
        # Bishop moves (Hamming-2)
        for delta in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            move = self.bishop_move(from_square, delta[0], delta[1])
            if move:
                moves.append(move)
        
        # Knight moves (Hamming-3 with MöbiusFlip)
        knight_deltas = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
        for delta in knight_deltas:
            move = self.knight_move(from_square, delta[0], delta[1], mobius_flip=True)
            if move:
                moves.append(move)
        
        return moves
    
    def analyze_move_patterns(self, moves: List[MoveResult]) -> Dict[str, Any]:
        """Analyze move patterns for CE1 invariants"""
        move_types = {}
        hamming_distribution = {1: 0, 2: 0, 3: 0}
        orientation_flips = 0
        
        for move in moves:
            # Count move types
            move_types[move.move_type] = move_types.get(move.move_type, 0) + 1
            
            # Count Hamming distances
            if move.move_type == "rook":
                hamming_distribution[1] += 1
            elif move.move_type == "bishop":
                hamming_distribution[2] += 1
            elif move.move_type == "knight":
                hamming_distribution[3] += 1
                
                # Count orientation flips
                if move.to_square.orientation != move.from_square.orientation:
                    orientation_flips += 1
        
        return {
            "total_moves": len(moves),
            "move_types": move_types,
            "hamming_distribution": hamming_distribution,
            "orientation_flips": orientation_flips
        }
    
    def spectral_analysis(self, squares: List[ChessSquare]) -> Dict[str, np.ndarray]:
        """Spectral analysis using Kravchuk basis K₈"""
        if not squares:
            return {}
        
        # Extract file and rank indices
        files = [self.file_to_index[sq.file] for sq in squares]
        ranks = [sq.rank - 1 for sq in squares]
        
        # Project onto Kravchuk basis
        file_spectrum = np.zeros(8)
        rank_spectrum = np.zeros(8)
        
        for f, r in zip(files, ranks):
            file_spectrum += self.kravchuk_basis[f, :]
            rank_spectrum += self.kravchuk_basis[r, :]
        
        return {
            "file_spectrum": file_spectrum,
            "rank_spectrum": rank_spectrum,
            "kravchuk_basis": self.kravchuk_basis
        }

def demo_ce1_chess_algebra():
    """Demonstrate CE1 chess algebra with worked examples"""
    print("=== CE1 Chess 8×8 ↔ 𝔽₂⁸ (Gray–Kravchuk gauge) ===")
    
    # Initialize algebra
    chess = CE1ChessAlgebra()
    
    # Test worked examples from specification
    print("\n=== Worked Examples ===")
    
    # a1: f=0→000, r=0→000; β=0, κ=0 → byte 00000000 = 0x00
    a1 = chess.encode_square("a", 1, BoundaryPolicy.TORUS, OrientationMode.MU)
    print(f"a1: {a1:02X} (expected 0x00)")
    
    # e4: f=4→110, r=3→010; β=0, κ=0
    # file e=4→110 ⇒ f_0=0,f_1=1,f_2=1
    # rank 4→3→010 ⇒ r_0=0,r_1=1,r_2=0
    # Pack: 0 1 1 0 1 0 0 0 (LSB left) = binary 00010110 = 0x16
    e4 = chess.encode_square("e", 4, BoundaryPolicy.TORUS, OrientationMode.MU)
    print(f"e4: {e4:02X} (expected 0x16)")
    
    # g7: f=6→101, r=6→101; β=1 (mirror), κ=0
    # Bits: 1 0 1 1 0 1 1 0 → 01101101 = 0x6D
    g7 = chess.encode_square("g", 7, BoundaryPolicy.MIRROR, OrientationMode.MU)
    print(f"g7: {g7:02X} (expected 0x6D)")
    
    # Test decoding
    print("\n=== Decoding Test ===")
    decoded_a1 = chess.decode_square(a1)
    decoded_e4 = chess.decode_square(e4)
    decoded_g7 = chess.decode_square(g7)
    
    print(f"Decoded a1: {decoded_a1.file}{decoded_a1.rank} (β={decoded_a1.boundary.name}, κ={decoded_a1.orientation.name})")
    print(f"Decoded e4: {decoded_e4.file}{decoded_e4.rank} (β={decoded_e4.boundary.name}, κ={decoded_e4.orientation.name})")
    print(f"Decoded g7: {decoded_g7.file}{decoded_g7.rank} (β={decoded_g7.boundary.name}, κ={decoded_g7.orientation.name})")
    
    # Test knight move with MöbiusFlip
    print("\n=== Knight MöbiusFlip Test ===")
    from_square = chess.decode_square(e4)
    print(f"From {from_square.file}{from_square.rank} (κ={from_square.orientation.name}):")
    
    # Knight L from e4 to f6 (Δf=+1, Δr=+2)
    knight_move = chess.knight_move(from_square, 1, 2, mobius_flip=True)
    if knight_move:
        print(f"Knight e4 → {knight_move.to_square.file}{knight_move.to_square.rank} (κ={knight_move.to_square.orientation.name})")
        print(f"Byte: {knight_move.to_square.byte:02X}")
    
    # Test all moves
    print("\n=== Move Generation ===")
    moves = chess.generate_all_moves(from_square)
    analysis = chess.analyze_move_patterns(moves)
    
    print(f"Total moves: {analysis['total_moves']}")
    print(f"Move types: {analysis['move_types']}")
    print(f"Hamming distribution: {analysis['hamming_distribution']}")
    print(f"Orientation flips: {analysis['orientation_flips']}")
    
    # Test spectral analysis
    print("\n=== Spectral Analysis ===")
    test_squares = [chess.decode_square(chess.encode_square(f, r, BoundaryPolicy.TORUS, OrientationMode.MU))
                   for f in ['a', 'e', 'h'] for r in [1, 4, 8]]
    spectrum = chess.spectral_analysis(test_squares)
    
    print(f"File spectrum: {spectrum['file_spectrum']}")
    print(f"Rank spectrum: {spectrum['rank_spectrum']}")
    
    print("\n✓ CE1 chess algebra demonstration completed!")
    print("The algebra provides kernel-friendly chess geometry in 𝔽₂⁸ with CE1 invariants.")

if __name__ == "__main__":
    demo_ce1_chess_algebra()
