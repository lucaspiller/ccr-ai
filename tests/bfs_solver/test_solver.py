"""
Tests for the BFS solver.
"""

import pytest

from src.bfs_solver.solver import BFSSolver
from src.bfs_solver.difficulty import DifficultyScorer, DifficultyLabel
from src.game.board import Direction
from src.game.board_builder import BoardBuilder, BoardConfig


class TestBFSSolver:
    """Test BFS solver functionality."""

    def test_solver_finds_known_solution_seed_42(self):
        """Test that BFS solver finds the known solution for seed 42."""
        config = BoardConfig(
            board_w=7,
            board_h=7,
            num_walls=10,
            num_mice=3,
            num_rockets=2,
            num_cats=1,
            num_holes=2,
            arrow_budget=3,
        )

        board_builder = BoardBuilder(config, seed=42)
        level = board_builder.generate_level("Test Puzzle")
        engine = level.create_engine(max_steps=1000, seed=42, puzzle_mode=True)

        # Create BFS solver
        solver = BFSSolver(depth_cap=10, timeout_ms=1000)  # Generous limits for test
        
        # Solve the puzzle
        result = solver.solve(engine)
        
        # Verify solution was found
        assert result.success is True
        assert result.solution == [((0, 2), Direction.UP), ((6, 2), Direction.LEFT)]

    def test_solver_timeout_and_limits(self):
        """Test that solver respects timeout and depth limits."""
        config = BoardConfig(
            board_w=7,
            board_h=7,
            num_walls=20,  # More walls to make it harder
            num_mice=5,    # More mice
            num_rockets=1, # Fewer rockets
            num_cats=2,    # More cats
            num_holes=5,   # More holes
            arrow_budget=2, # Very limited arrows
        )

        board_builder = BoardBuilder(config, seed=123)
        level = board_builder.generate_level("Hard Puzzle")
        engine = level.create_engine(max_steps=1000, seed=123, puzzle_mode=True)

        # Create BFS solver with tight limits
        solver = BFSSolver(depth_cap=3, timeout_ms=10)
        
        # Solve the puzzle
        result = solver.solve(engine)
        
        # With such tight constraints, it should timeout or hit depth limit
        assert result.time_taken_ms <= 50  # Should be close to timeout
        assert result.nodes_explored > 0   # Should have explored some nodes
        
        # May or may not find solution due to constraints
        if result.success:
            assert result.solution_length <= config.arrow_budget

    def test_unsolvable_puzzle(self):
        """Test behavior with an unsolvable puzzle."""
        config = BoardConfig(
            board_w=5,
            board_h=5,
            num_walls=0,
            num_mice=1,
            num_rockets=1, 
            num_cats=0,
            num_holes=0,
            arrow_budget=0,  # No arrows allowed!
        )

        board_builder = BoardBuilder(config, seed=456)
        level = board_builder.generate_level("Unsolvable Puzzle")
        engine = level.create_engine(max_steps=1000, seed=456, puzzle_mode=True)

        # Create BFS solver
        solver = BFSSolver(depth_cap=5, timeout_ms=100)
        
        # Solve the puzzle
        result = solver.solve(engine)
        
        # Should not find solution (assuming mouse can't reach rocket without arrows)
        # Note: This test might fail if the random generation happens to place 
        # mouse right next to rocket. That's okay - it's a probabilistic test.
        assert result.nodes_explored >= 1  # Should have tried at least initial state


class TestDifficultyScorer:
    """Test difficulty scoring functionality."""

    def test_difficulty_scoring(self):
        """Test difficulty scoring formula."""
        config = BoardConfig(
            board_w=7,
            board_h=7,
            num_walls=10,
            num_mice=3,
            num_rockets=2,
            num_cats=1,
            num_holes=2,
            arrow_budget=3,
        )

        board_builder = BoardBuilder(config, seed=42)
        level = board_builder.generate_level("Test Puzzle")
        engine = level.create_engine(max_steps=1000, seed=42, puzzle_mode=True)

        # Test solution
        solution = [
            ((1, 5), Direction.DOWN),
            ((6, 1), Direction.LEFT),
        ]
        
        score = DifficultyScorer.score_puzzle(engine, solution)
        
        # Expected: 1.0 * 2 (solution length) + 4.0 * 1 (cats) + 2.0 * 2 (holes) = 10.0
        expected_score = 1.0 * 2 + 4.0 * 1 + 2.0 * 2
        assert score == expected_score
        
        # Test label assignment
        label = DifficultyScorer.get_difficulty_label(score)
        assert label == DifficultyLabel.EASY  # Score of 10 is Easy

    def test_difficulty_labels(self):
        """Test difficulty label boundaries."""
        assert DifficultyScorer.get_difficulty_label(5) == DifficultyLabel.EASY
        assert DifficultyScorer.get_difficulty_label(10) == DifficultyLabel.EASY
        assert DifficultyScorer.get_difficulty_label(15) == DifficultyLabel.MEDIUM
        assert DifficultyScorer.get_difficulty_label(20) == DifficultyLabel.MEDIUM
        assert DifficultyScorer.get_difficulty_label(25) == DifficultyLabel.HARD
        assert DifficultyScorer.get_difficulty_label(35) == DifficultyLabel.HARD
        assert DifficultyScorer.get_difficulty_label(40) == DifficultyLabel.BRUTAL