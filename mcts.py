import math
import random
import torch
import torch.nn.functional as F
import io
import chess
import chess.pgn
from chess.engine import SimpleEngine, Limit
from typing import List, Optional

STOCKFISH_PATH = "/root/chess-hackathon/utils/stockfish"


class MCTSNode:
    def __init__(self, pgn, device, parent=None, move=None):
        self.pgn = pgn
        self.parent = parent
        self.move = move
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.total_score = torch.tensor(0.0, device=device)
        self.device = device

    def add_child(self, child_pgn: str, move: str) -> "MCTSNode":
        """
        Create and add a child node to this node

        Args:
            child_pgn (str): PGN string for the child node
            move (str): Move that led to this child node

        Returns:
            MCTSNode: The newly created child node
        """
        child_node = MCTSNode(pgn=child_pgn, device=self.device, parent=self, move=move)
        self.children.append(child_node)
        return child_node

    def uct_score(self, exploration_weight: float = 1.0) -> float:
        """
        Calculate the Upper Confidence Bound (UCT) score for this node

        Args:
            exploration_weight (float): Controls exploration vs exploitation

        Returns:
            float: UCT score
        """
        # Handle the case of unvisited node
        if self.visits == 0:
            return float("inf")

        # Ensure we're working with scalar values
        parent_visits = self.parent.visits if self.parent else 1
        visits = max(self.visits, 1)

        # Compute exploitation term (average score)
        exploitation_term = (self.total_score / visits).item()

        # Compute exploration term
        exploration_term = exploration_weight * math.sqrt(
            math.log(parent_visits) / visits
        )

        return exploitation_term + exploration_term


class MCTS:
    def __init__(self, model, timer, device, num_simulations: int = 50):
        """
        Initialize Monte Carlo Tree Search

        Args:
            model: Scoring model for evaluating positions
            device: Computational device (CPU/GPU)
            num_simulations: Number of MCTS simulations per move
        """
        self.device = device
        self.model = model
        self.num_simulations = num_simulations
        self.stockfish_engine = None
        self.timer = timer

    def _uci_to_san(self, board: chess.Board, uci_move: str) -> str:
        """
        Convert a UCI move to Standard Algebraic Notation (SAN)

        Args:
            board (chess.Board): Current chess board state
            uci_move (str): Move in UCI format

        Returns:
            str: Move in Standard Algebraic Notation
        """
        try:
            move = chess.Move.from_uci(uci_move)
            if not board.is_legal(move):
                legal_moves = list(board.legal_moves)
                # Pick a random legal move
                move = random.choice(legal_moves)  
                # print(f"Illegal move: {uci_move}. Choosing random legal move: {move.uci()}")
            return board.san(move)
        except Exception as e:
            print(f"Error converting UCI to SAN:\n{e}")
            # print(f"Board:\n{board}")
            # print(f"uci_move:\n{uci_move}")
            return uci_move

    def _get_stockfish_engine(self) -> SimpleEngine:
        """
        Lazily initialize Stockfish engine

        Returns:
            SimpleEngine: Stockfish chess engine
        """
        if self.stockfish_engine is None:
            self.stockfish_engine = SimpleEngine.popen_uci(STOCKFISH_PATH)
        return self.stockfish_engine

    def _fallback_move(self, pgn: str) -> Optional[str]:
        """
        Generate a fallback legal move when MCTS fails

        Args:
            pgn (str): Game PGN

        Returns:
            Optional[str]: UCI move or None
        """
        try:
            game = chess.pgn.read_game(io.StringIO(pgn))
            board = game.board()
            legal_moves = list(board.legal_moves)
            return legal_moves[0].uci() if legal_moves else None
        except Exception as e:
            print(f"Fallback move generation error: {e}")
            return None

    def select_best_moves(self, pgn_batch: List[str]) -> List[Optional[str]]:
        """
        Select best moves for a batch of PGNs

        Args:
            pgn_batch (List[str]): Batch of PGN strings

        Returns:
            List[Optional[str]]: Best moves for each PGN
        """
        best_moves = []
        for pgn in pgn_batch:
            try:
                # Get the current board state
                game = chess.pgn.read_game(io.StringIO(pgn))
                board = game.board()
                # Select best move
                uci_best_move = self.select_best_move(pgn)
                # Convert UCI to SAN
                if uci_best_move:
                    san_best_move = self._uci_to_san(board, uci_best_move)
                    best_moves.append(san_best_move)
                else:
                    best_moves.append(None)
            except Exception as e:
                print(f"Move selection error for PGN {pgn}: {e}")
                best_moves.append(self._fallback_move(pgn))
        return best_moves

    def select_best_move(self, current_pgn: str) -> Optional[str]:
        """
        Select the best move for a given PGN using MCTS

        Args:
            current_pgn (str): Current game state PGN

        Returns:
            Optional[str]: Best move in UCI format
        """
        root = MCTSNode(current_pgn, device=self.device)

        for _ in range(self.num_simulations):
            # Selection phase
            leaf_node = self._select(root)

            # Expansion phase
            expanded_node = self._expand(leaf_node)

            # Simulation phase
            if expanded_node:
                score = self._simulate(expanded_node)

                # Backpropagation phase
                self._backpropagate(expanded_node, score)

        # Choose the best child based on visits
        if not root.children:
            print("No moves generated. Using fallback.")
            return self._fallback_move(current_pgn)

        best_child = max(root.children, key=lambda child: child.visits)
        return best_child.move

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Select the most promising node for expansion

        Args:
            node (MCTSNode): Starting node for selection

        Returns:
            MCTSNode: Selected leaf node
        """
        while node.children:
            # If no children can be selected, break
            if not node.children:
                break

            # Select child with best UCT score
            node = max(node.children, key=lambda child: child.uct_score())

        return node

    def _expand(self, node: MCTSNode) -> Optional[MCTSNode]:
        """
        Expand the selected node by generating child nodes

        Args:
            node (MCTSNode): Node to expand

        Returns:
            Optional[MCTSNode]: First expanded child node
        """
        # Generate move candidates
        try:
            move_candidates = self._generate_move_candidates(node.pgn)
        except Exception as e:
            print(f"Move generation error: {e}")
            return None

        # If no move candidates, return None
        if not move_candidates:
            return None

        # Create child nodes for each move candidate
        for move in move_candidates:
            new_pgn = f"{node.pgn} {move}"
            node.add_child(new_pgn, move)

        # Return the first child node
        return node.children[0] if node.children else None

    def _generate_move_candidates(
        self,
        pgn: str,
        depth_limit: int = 5,
        time_limit: int = 1,
        topk: Optional[int] = None,
    ) -> List[str]:
        """
        Generate candidate moves for a given PGN

        Args:
            pgn (str): Current game state PGN
            depth_limit (int): Stockfish search depth
            time_limit (int): Stockfish time limit
            topk (Optional[int]): Limit number of candidate moves

        Returns:
            List[str]: List of UCI move candidates
        """
        # Use Stockfish to generate candidate moves
        engine = self._get_stockfish_engine()
        game = chess.pgn.read_game(io.StringIO(pgn))
        board = game.board()

        # Push existing moves to the board
        for move in game.mainline_moves():
            board.push(move)

        # Get legal moves
        legal_moves = list(board.legal_moves)

        # Convert to UCI format and limit if needed
        move_candidates = [move.uci() for move in legal_moves]
        return move_candidates[:topk] if topk else move_candidates

    def _simulate(self, node: MCTSNode) -> float:
        """
        Simulate the value of a node

        Args:
            node (MCTSNode): Node to evaluate

        Returns:
            float: Simulation score
        """
        try:
            return self.model.module.score(node.pgn, node.move)
        except Exception as e:
            print(f"Simulation error: {e}")
            return 0.0

    def _backpropagate(self, node: MCTSNode, score: float):
        """
        Backpropagate the simulation score

        Args:
            node (MCTSNode): Starting node for backpropagation
            score (float): Simulation score
        """
        score_tensor = torch.tensor(score, device=self.device)

        while node:
            node.visits += 1
            node.total_score += score_tensor
            node = node.parent

    def __del__(self):
        """
        Clean up Stockfish engine on object deletion
        """
        if hasattr(self, "stockfish_engine") and self.stockfish_engine:
            try:
                self.stockfish_engine.quit()
            except Exception as e:
                print(f"Error closing Stockfish engine: {e}")
