import sys
import time
import re
from copy import deepcopy

# 增加遞歸深度以防萬一
sys.setrecursionlimit(5000)

# 常量設定
UNKNOWN = -1
EMPTY = 0
FILLED = 1

# 給每題的預算時間 (秒)
MAX_TIME_PER_PUZZLE = 15.0

class NonogramSolver:
    def __init__(self, col_hints, row_hints, pid="Unknown"):
        self.N = 25
        self.col_hints = col_hints
        self.row_hints = row_hints
        self.pid = pid
        self.grid = [[UNKNOWN] * self.N for _ in range(self.N)]
        self.start_time = 0

    def solve(self):
        self.start_time = time.time()
        
        # 1. 邏輯推導 (Constraint Propagation)
        # 反覆進行直到無法推導出新資訊
        if not self._propagate():
            return None # 矛盾
            
        # 2. 如果推導完還有未知格子，進入回溯 (Backtracking)
        if self._is_solved():
            return self.grid
        
        if self._backtrack():
            return self.grid
            
        return None

    def _is_solved(self):
        for r in range(self.N):
            for c in range(self.N):
                if self.grid[r][c] == UNKNOWN:
                    return False
        return True

    def _propagate(self):
        """反覆對每一行/列進行邏輯解鎖，直到穩定"""
        changed = True
        while changed:
            if time.time() - self.start_time > MAX_TIME_PER_PUZZLE:
                return True # 超時，保留現有進度進入猜測階段
            
            changed = False
            
            # --- Check Rows ---
            for r in range(self.N):
                # 呼叫核心解算器
                new_line = self._solve_line_complete(self.grid[r], self.row_hints[r])
                if new_line is None:
                    # Debug Info
                    # print(f"[{self.pid}] Contradiction at Row {r+1}")
                    return False
                if new_line != self.grid[r]:
                    self.grid[r] = new_line
                    changed = True
            
            # --- Check Cols ---
            for c in range(self.N):
                current_col = [self.grid[r][c] for r in range(self.N)]
                new_col = self._solve_line_complete(current_col, self.col_hints[c])
                if new_col is None:
                    # Debug Info
                    # print(f"[{self.pid}] Contradiction at Col {c+1}")
                    return False
                if new_col != current_col:
                    for r in range(self.N):
                        self.grid[r][c] = new_col[r]
                    changed = True
        return True

    # =========================================================================
    # 核心演算法：遞歸全排列檢查 (Recursive Permutation Checker)
    # =========================================================================
    def _solve_line_complete(self, line, hints):
        """
        找出所有符合 'line' 現狀 (UNKNOWN/FILLED/EMPTY) 且符合 'hints' 的排列。
        並返回所有合法排列的 '交集' (Commonality)。
        """
        N = self.N
        # 預檢查：空間是否足夠
        if not hints:
            # 如果沒有 hint，線必須全白
            for x in line:
                if x == FILLED: return None
            return [EMPTY] * N
            
        min_len = sum(hints) + len(hints) - 1
        if min_len > N: return None

        # 準備遞歸用的 Cache
        memo = {}
        
        # 我們需要統計對於每個位置 i，所有合法排列中該位置是 0 還是 1
        # 使用位元運算或計數器
        # can_be_0[i] = True/False
        # can_be_1[i] = True/False
        can_be_0 = [False] * N
        can_be_1 = [False] * N
        
        # 標記是否找到至少一個合法解
        found_any_solution = [False] 

        def recursive_search(hint_idx, start_pos):
            # 狀態：正在放置第 hint_idx 個 block，搜尋起點為 start_pos
            state = (hint_idx, start_pos)
            if state in memo: return memo[state]
            
            # --- Base Case: 所有 Block 都放完了 ---
            if hint_idx == len(hints):
                # 檢查剩餘空間是否可以全填空白
                # 從 start_pos 到 N-1 必須不能有 FILLED
                for k in range(start_pos, N):
                    if line[k] == FILLED:
                        memo[state] = False
                        return False
                
                # 找到一個合法排列！
                found_any_solution[0] = True
                # 回溯並標記路徑上的空白
                for k in range(start_pos, N):
                    can_be_0[k] = True
                return True

            # --- Pruning: 空間不足 ---
            remaining_len = sum(hints[hint_idx:]) + (len(hints) - 1 - hint_idx)
            if start_pos + remaining_len > N:
                memo[state] = False
                return False

            current_block_len = hints[hint_idx]
            # Block 可以放置的範圍：從 start_pos 到 N - remaining_len
            limit = N - remaining_len + 1
            
            has_valid_path = False
            
            for pos in range(start_pos, limit):
                # 嘗試將 block 放在 [pos, pos + len)
                
                # 1. 檢查前置空白 (從 start_pos 到 pos)
                # 這些格子必須是 EMPTY，所以不能是 FILLED
                gap_valid = True
                for k in range(start_pos, pos):
                    if line[k] == FILLED:
                        gap_valid = False
                        break
                if not gap_valid:
                    # 如果這裡不能留白，那更後面的位置也不可能，因為必須跳過這個 FILLED
                    # 這一點非常重要：如果我們跳過了一個 FILLED，後面的 block 也不可能回頭蓋住它
                    # 所以可以直接 break loop
                    break 

                # 2. 檢查 Block 本身 (從 pos 到 pos + len)
                # 這些格子必須是 FILLED，所以不能是 EMPTY
                block_valid = True
                for k in range(pos, pos + current_block_len):
                    if line[k] == EMPTY:
                        block_valid = False
                        break
                
                # 3. 檢查 Block 後的分隔 (pos + len)
                # 必須是 EMPTY，所以不能是 FILLED
                if block_valid and (pos + current_block_len < N):
                    if line[pos + current_block_len] == FILLED:
                        block_valid = False
                
                if block_valid:
                    # 嘗試遞歸
                    next_start = pos + current_block_len + 1
                    if recursive_search(hint_idx + 1, next_start):
                        has_valid_path = True
                        
                        # --- 記錄可行性 ---
                        # 路徑上的前置空白
                        for k in range(start_pos, pos):
                            can_be_0[k] = True
                        # Block 本身
                        for k in range(pos, pos + current_block_len):
                            can_be_1[k] = True
                        # 分隔空白
                        if pos + current_block_len < N:
                            can_be_0[pos + current_block_len] = True

            memo[state] = has_valid_path
            return has_valid_path

        # 啟動遞歸
        recursive_search(0, 0)
        
        if not found_any_solution[0]:
            return None # 無解，矛盾

        # 整合結果
        new_line = list(line)
        for i in range(N):
            if can_be_1[i] and not can_be_0[i]:
                new_line[i] = FILLED
            elif can_be_0[i] and not can_be_1[i]:
                new_line[i] = EMPTY
            # 如果兩者皆 True，保持 UNKNOWN
            
        return new_line

    def _backtrack(self):
        # 簡單啟發式：找最少未知數的位置，或直接找第一個未知
        best_r, best_c = -1, -1
        
        for r in range(self.N):
            for c in range(self.N):
                if self.grid[r][c] == UNKNOWN:
                    best_r, best_c = r, c
                    break
            if best_r != -1: break
        
        if best_r == -1: return True

        # 備份狀態
        saved_grid = deepcopy(self.grid)

        # 嘗試填 1
        self.grid[best_r][best_c] = FILLED
        if self._propagate():
            if self._backtrack(): return True
        
        # 回溯 -> 填 0
        self.grid = saved_grid
        self.grid[best_r][best_c] = EMPTY
        if self._propagate():
            if self._backtrack(): return True
            
        return False

# ======================================================
# 智慧解析器 (Smart Parser)
# ======================================================
def is_valid_hint(hints, N=25):
    """檢查一組 hint 是否在 N 格內合法"""
    if not hints: return True # 空 hint (全白) 合法
    # 最小所需長度 = 總和 + (區塊數 - 1)
    min_len = sum(hints) + len(hints) - 1
    return min_len <= N

def parse_input_file(filename):
    """
    讀取並解析輸入檔，返回一個 dict: {pid: (col_hints, row_hints)}
    """
    puzzles = {}
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f]
    except FileNotFoundError:
        print("Input file not found.")
        return {}

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        
        # 1. 尋找題號
        if '$' in line:
            # 提取題號，例如 $1
            pid_match = re.search(r'\$(\d+)', line)
            if not pid_match:
                idx += 1
                continue
            
            pid = f"${pid_match.group(1)}"
            # print(f"Found Puzzle {pid}")
            idx += 1
            
            col_hints = []
            row_hints = []
            
            # 2. 讀取 50 行有效的 Hints (25 Cols + 25 Rows)
            # 我們需要跳過雜訊行 (如 , 空行, 或明顯不合法的行)
            hints_buffer = []
            
            while len(hints_buffer) < 50 and idx < len(lines):
                curr_line = lines[idx]
                idx += 1
                
                # 跳過空行
                if not curr_line: continue
                
                # 跳過明確的 source 標記行 (這是關鍵修正!)
                if '[source' in curr_line or 'source:' in curr_line:
                    continue

                # 解析數字
                # 移除行內可能的註釋或非數字字符
                clean_line = curr_line.split('#')[0]
                numbers = re.findall(r'\d+', clean_line)
                hints = [int(x) for x in numbers]
                
                # 檢查合法性 (物理限制)
                if is_valid_hint(hints, 25):
                    hints_buffer.append(hints)
                else:
                    # 如果讀到不合法的 hint，可能是讀到了雜訊，忽略
                    # print(f"Skipping invalid hint line: {curr_line}")
                    pass
            
            if len(hints_buffer) == 50:
                col_hints = hints_buffer[:25]
                row_hints = hints_buffer[25:]
                puzzles[pid] = (col_hints, row_hints)
            else:
                print(f"Warning: Puzzle {pid} has incomplete hints ({len(hints_buffer)}/50).")
        else:
            idx += 1
            
    return puzzles

# ======================================================
# 主程式 (Main)
# ======================================================
def main():
    input_file = 'test_input.txt' 
    output_file = 'result_py2.txt'
    
    # 使用新解析器一次讀取所有題目
    print(f"Parsing {input_file}...")
    all_puzzles = parse_input_file(input_file)
    print(f"Loaded {len(all_puzzles)} puzzles.")

    with open(output_file, 'w') as f_out:
        # 按照題號順序處理 (雖然 dict 順序通常保留，但為了保險)
        # 假設題號是 $1, $2... $10
        # 為了處理非數字順序，我們直接遍歷讀到的 pid
        
        for pid, (col_hints, row_hints) in all_puzzles.items():
            print(f"Solving {pid}...")
            
            solver = NonogramSolver(col_hints, row_hints, pid)
            sol = solver.solve()
            
            f_out.write(f"{pid}\n")
            if sol:
                print(f"-> Solved {pid}!")
                for row in sol:
                    f_out.write("  ".join(["1" if x==1 else "0" for x in row]) + "\n")
            else:
                print(f"-> {pid} Unsolvable.")
                f_out.write("Unsolvable\n")
            
            f_out.flush()

if __name__ == "__main__":
    main()
# ======================================================
# 主程式 (Main)
# ======================================================
def main():
    input_file = 'test_input.txt' 
    output_file = 'result_py2.txt'
    
    # 使用新解析器一次讀取所有題目
    print(f"Parsing {input_file}...")
    all_puzzles = parse_input_file(input_file)
    print(f"Loaded {len(all_puzzles)} puzzles.")

    with open(output_file, 'w') as f_out:
        # 按照題號順序處理 (雖然 dict 順序通常保留，但為了保險)
        # 假設題號是 $1, $2... $10
        # 為了處理非數字順序，我們直接遍歷讀到的 pid
        
        for pid, (col_hints, row_hints) in all_puzzles.items():
            print(f"Solving {pid}...")
            
            solver = NonogramSolver(col_hints, row_hints, pid)
            sol = solver.solve()
            
            f_out.write(f"{pid}\n")
            if sol:
                print(f"-> Solved {pid}!")
                for row in sol:
                    f_out.write("\t".join(["1" if x==1 else "0" for x in row]) + "\n")
            else:
                print(f"-> {pid} Unsolvable.")
                f_out.write("Unsolvable\n")
            
            f_out.flush()

if __name__ == "__main__":
    main()