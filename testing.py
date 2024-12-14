pgn_batch = ['1.d4 g6 2.e4 Bg7 3.c4 c5 4.d5 d6 5.Nf3 Nd7 6.Nc3 Bxc3+ 7.bxc3 Ndf6 8.Bd3 e5 9.dxe6 Bxe6 10.Rb1 Qc7 11.O-O h6 12.Ne1 O-O-O 13.Nc2 Ne7 14.Ne3 Nc6 15.Nd5 Bxd5 16.cxd5 Ne5 17.f3 g5 18.Rf2 g4 19.Rfb2 c4 20.Be2 gxf3 21.gxf3 b6 22.Be3 Nfd7 23.a4 f5 24.exf5 Rde8 25.Bd4 Nc5 26.Kh1 Rhg8 27.a5 Ned3 28.Bxd3 Nxd3 29.Re2 Qb7 30.axb6 axb6 31.Rxe8+ Rxe8 32.Qa4 b5 33.Rxb5 Re1+ 34.Kg2 Nf4+ 35.Kg2 Nf4+  1-0']
mcts_moves = ['g8h6']

def augment_pgn_with_mcts(pgn_batch, mcts_moves):
    augmented_pgn_batch = []
    for pgn, move in zip(pgn_batch, mcts_moves):
        # Split the PGN to get the moves and the result
        moves_and_result = pgn.split('{')[0]
        moves_and_result = moves_and_result.strip()
        moves_and_result = moves_and_result.split(' ')
        result = moves_and_result[-1]
        # All moves except the result
        moves = ' '.join(moves_and_result[:-1])  
        moves = moves.strip()
        # To determine the current move number
        # Each move is split by space, so count half for the move number
        move_number = len(moves.split()) // 2
        # Insert the MCTS move with the correct move number
        mcts_move_with_number = f"{move_number + 1}.{move}"
        # Construct the augmented PGN: add the MCTS move and then append the result
        augmented_pgn = f"{moves} {mcts_move_with_number} {result}"
        augmented_pgn_batch.append(augmented_pgn)
    
    return augmented_pgn_batch


x = augment_pgn_with_mcts(pgn_batch, mcts_moves)
print(x)
