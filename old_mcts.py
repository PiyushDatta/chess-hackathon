import math
import random
import torch
import torch.nn.functional as F
import io
import chess.pgn
from chess import Board
from chess.engine import SimpleEngine, Limit
from tqdm import tqdm
from utils.data_utils import score_possible_boards

STOCKFISH_PATH = "/root/chess-hackathon/utils/stockfish"


class MCTSNode:
    def __init__(self, pgn, device, parent=None, move=None):
        self.pgn = pgn
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.total_score = torch.tensor(0.0, device=device)
        self.device = device

    def add_child(self, child_pgn, move):
        child_node = MCTSNode(child_pgn, device=self.device, parent=self, move=move)
        self.children.append(child_node)
        return child_node

    def uct_score(self, exploration_weight=1.0):
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits
        visits = self.visits
        # Check if both are scalars
        if isinstance(parent_visits, torch.Tensor):
            parent_visits = parent_visits.item()
        if isinstance(visits, torch.Tensor):
            visits = visits.item()
        # Compute the exploration term
        exploration_term = exploration_weight * torch.sqrt(
            torch.log(
                torch.tensor(parent_visits, dtype=torch.float32, device=self.device)
            )
            / visits
        )
        return (self.total_score / visits) + exploration_term.item()


class MCTS:
    def __init__(self, model, timer, device, num_simulations=50):
        self.device = device
        self.model = model
        self.num_simulations = num_simulations
        self.timer = timer
        self.stockfish_engine = None
        self.move_cache = {}

    def __del__(self):
        if self.stockfish_engine:
            self.stockfish_engine.quit()

    def _get_stockfish_engine(self):
        if not self.stockfish_engine:
            self.stockfish_engine = SimpleEngine.popen_uci("/root/chess-hackathon/utils/stockfish")
        return self.stockfish_engine

    def select_best_moves(self, pgn_batch):
        """
        Select best moves for an entire batch of PGNs
        """
        best_moves = []
        for pgn in pgn_batch:
            try:
                best_move = self.select_best_move(pgn)
                best_moves.append(best_move)
            except Exception as e:
                print(f"Error selecting best move for PGN {pgn}: {e}")
                # Fallback: choose a random legal move
                best_move = self._fallback_move(pgn)
                best_moves.append(best_move)
        return best_moves
        # best_moves = []
        # for pgn in pgn_batch:
        #     # Extract the initial state or first few moves
        #     best_move = self.select_best_move(pgn)
        #     best_moves.append(best_move)
        # return best_moves

    def _fallback_move(self, pgn):
        """
        Fallback method to generate a legal move when MCTS fails
        """
        # Parse the PGN and get the board state
        game = chess.pgn.read_game(io.StringIO(pgn))
        board = game.board()
        
        # Get all legal moves
        legal_moves = list(board.legal_moves)
        
        # Return a random legal move or None if no moves
        return legal_moves[0].uci() if legal_moves else None

    def select_best_move(self, current_pgn):
        root = {
            'pgn': current_pgn,
            'children': [],
            'visits': 0,
            'total_score': 0.0
        }
        # Preallocate children to reduce memory churn
        root['children'] = [None] * 20  # Limit to 20 potential moves
        for _ in range(self.num_simulations):
            node = self._select(root)
            expanded_node = self._expand(node)
            score = self._simulate(expanded_node)
            self._backpropagate(expanded_node, score)
        # Find best move using visits
        best_child = max(
            (child for child in root['children'] if child),
            key=lambda x: x['visits'],
            default=None
        )
        return best_child['move'] if best_child else None
        # root = MCTSNode(current_pgn, device=self.device)

        # for _ in range(self.num_simulations):
        #     # Selection
        #     node = self._select(root)
        #     # Expansion
        #     expanded_node = self._expand(node)
        #     # Simulation (Rollout)
        #     score = self._simulate(expanded_node)
        #     # Backpropagation
        #     self._backpropagate(expanded_node, score)

        # # Choose the best child based on visits
        # if not root.children:
        #     return None  # No moves found
        # best_child = max(root.children, key=lambda child: child.visits)
        # return best_child.move

    def _select(self, node):
        while node['children']:
            node = max(
                (child for child in node['children'] if child),
                key=lambda child: (
                    child['total_score'] / (child['visits'] + 1e-5) + 
                    math.sqrt(math.log(node['visits'] + 1) / (child['visits'] + 1e-5))
                )
            )
        return node
        # Select child with best UCT score
        # while node.children:
        #     node = max(node.children, key=lambda child: child.uct_score())
        # return node

    def _expand(self, node):
        cache_key = node['pgn']
        if cache_key not in self.move_cache:
            self.move_cache[cache_key] = self._generate_move_candidates(node['pgn'])
        moves = self.move_cache[cache_key]
        child = {
            'pgn': node['pgn'] + " " + moves[0],
            'move': moves[0],
            'children': [],
            'visits': 0,
            'total_score': 0.0,
            'parent': node
        }
        # Add to parent's children
        node['children'][0] = child
        return child
        # Generate possible moves
        # move_candidates = self._generate_move_candidates(node.pgn)
        # # Expand children with these moves
        # for move in move_candidates:
        #     new_pgn = node.pgn + " " + move
        #     node.add_child(new_pgn, move)
        # # Return first child if available
        # return node.children[0] if node.children else node

    def _generate_move_candidates(self, pgn, max_candidates=10):
        engine = self._get_stockfish_engine()
        board = chess.Board()
        # Parse existing moves
        moves = pgn.split()[1::2]  # Skip move numbers
        for move_san in moves:
            board.push_san(move_san)
        # Get legal moves efficiently
        legal_moves = list(board.legal_moves)
        # Limit and convert to UCI
        return [move.uci() for move in legal_moves[:max_candidates]]

    def _old_generate_move_candidates(
        self, pgn, depth_limt=5, time_limit=1, topk=None, verbose=False
    ):
        """
        Accepts pgn in the following format (or a format that can be parsed as follows):
        "1.Nc3 d5 2.d4 c6 3.a3 Nf6 4.Bf4 Bf5 5.Nf3 Nh5 6.h4 Nxf4 7.h5 0-1 {OL: 0}"
        Returns array of scored boards (N, 8, 8) and board scores (N,)
        """
        engine = SimpleEngine.popen_uci(STOCKFISH_PATH)
        game = chess.pgn.read_game(io.StringIO(pgn))
        board = Board()
        move_candidates = []

        # Evaluate the initial position
        _legal_moves, _legal_move_sans, possible_boards, scores = score_possible_boards(
            board, engine, depth_limt=depth_limt, time_limit=time_limit, topk=topk
        )
        for move in (
            tqdm(list(game.mainline_moves()))
            if verbose
            else list(game.mainline_moves())
        ):
            board.push(move)
            _legal_moves, _legal_move_sans, possible_boards, scores = (
                score_possible_boards(
                    board,
                    engine,
                    depth_limt=depth_limt,
                    time_limit=time_limit,
                    topk=topk,
                )
            )
            move_candidates.extend([move.uci() for move in _legal_moves])
        # return np.array(scored_boards), np.array(board_scores)
        # board = chess.Board(pgn)
        # game = chess.pgn.read_game(io.StringIO(pgn))
        # board = game.board()
        # move_candidates = [move.uci() for move in _legal_moves]
        return move_candidates
        # # Extract the last move or game state
        # last_move = pgn.split()[-1] if pgn else ""
        # # Generate some arbitrary move candidates
        # # In a real implementation, this should use chess rules
        # candidates = []
        # for piece in ["N", "B", "R", "Q", "K"]:
        #     for file in "abcdefgh":
        #         for rank in "12345678":
        #             move = f"{piece}{file}{rank}"
        #             candidates.append(move)

        # return candidates[:10]  # Limit the number of candidates

    def _simulate(self, node):
        """
        Simulate by evaluating the move using the model's scoring method.
        """
        try:
            score = self.model.module.score(node['pgn'], node['move'])
            # score = self.model.module.score(node.pgn, node.move)
            # print("SCORE:")
            # print(score)
            # exit(1)
            return score
        except Exception as e:
            print(f"Simulation error: {e}")
            return 0.0

    def _backpropagate(self, node, score):
        while node:
            node['visits'] += 1
            node['total_score'] += score
            node = node.get('parent')
        # score = torch.tensor(
        #     score, device=self.device
        # )
        # while node:
        #     node.visits += 1
        #     node.total_score += score
        #     node = node.parent
