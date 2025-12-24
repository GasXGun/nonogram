import sys
import time
import re
import multiprocessing
from copy import deepcopy

# 設定遞歸深度
sys.setrecursionlimit(50000)

# 常量
UNKNOWN = -1
EMPTY = 0
FILLED = 1

# 單題限時 (秒)
# 既然格式確定，解題應該會很快，600秒非常充裕
MAX_TIME_PER_PUZZLE = 600.0 

# ======================================================
# 核心解題類別 (Fast Array-Based DP)
# ======================================================
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
        if not self._propagate_queue(): return None 
        # 2. 檢查是否完成
        if self._is_solved(): return self.grid
        # 3. 回溯猜測 (Backtracking)
        if self._backtrack(): return self.grid
        return None

    def _is_solved(self):
        for r in range(self.N):
            for c in range(self.N):
                if self.grid[r][c] == UNKNOWN: return False
        return True

    def _propagate_queue(self):
        # 優先佇列：存儲需要檢查的 (type, index)
        # type 0: Row, type 1: Col
        queue = []
        for r in range(self.N): queue.append((0, r))
        for c in range(self.N): queue.append((1, c))
        in_queue = set(queue)

        while queue:
            if time.time() - self.start_time > MAX_TIME_PER_PUZZLE: return True
            
            q_type, idx = queue.pop(0)
            in_queue.remove((q_type, idx))

            if q_type == 0: # Row
                current = self.grid[idx]
                hints = self.row_hints[idx]
                new_line = self._solve_line_fast(current, hints)
                
                if new_line is None: return False # 矛盾
                if new_line != current:
                    self.grid[idx] = new_line
                    # 此行有變動，將相關的 Column 加入佇列
                    for c in range(self.N):
                        if current[c] != new_line[c]:
                            if (1, c) not in in_queue:
                                queue.append((1, c))
                                in_queue.add((1, c))
            else: # Col
                current = [self.grid[r][idx] for r in range(self.N)]
                hints = self.col_hints[idx]
                new_line = self._solve_line_fast(current, hints)
                
                if new_line is None: return False
                if new_line != current:
                    # 此列有變動，將相關的 Row 加入佇列
                    for r in range(self.N):
                        if self.grid[r][idx] != new_line[r]:
                            self.grid[r][idx] = new_line[r]
                            if (0, r) not in in_queue:
                                queue.append((0, r))
                                in_queue.add((0, r))
        return True

    # --- 高速陣列化 DP (比字典快) ---
    def _solve_line_fast(self, line, hints):
        N = self.N
        K = len(hints)
        
        # 快速檢查：總長度是否超過 N
        min_len_req = sum(hints) + K - 1
        if min_len_req > N: return None
        
        # 如果提示為空 (或只有0)，則整行必須為白
        if K == 0 or (K == 1 and hints[0] == 0):
            for x in line:
                if x == FILLED: return None
            return [EMPTY] * N

        # 預計算後綴長度
        suffix_len = [0] * (K + 1)
        for i in range(K - 1, -1, -1):
            suffix_len[i] = suffix_len[i+1] + hints[i] + (1 if i < K - 1 else 0)

        # memo[hint_index][current_pos]
        # 0: 未計算, 1: 可行, 2: 不可行
        memo = [[0] * (N + 2) for _ in range(K + 1)]
        
        # 用於合成結果的標記陣列
        pos_black = [False] * N
        pos_white = [False] * N

        def dp(h_idx, pos):
            if memo[h_idx][pos] != 0:
                return memo[h_idx][pos] == 1
            
            # Base Case: 所有提示已放入
            if h_idx == K:
                # 檢查剩餘格子是否全白
                for i in range(pos, N):
                    if line[i] == FILLED:
                        memo[h_idx][pos] = 2
                        return False
                # 成功路徑：標記路徑上的空白
                for i in range(pos, N): pos_white[i] = True
                memo[h_idx][pos] = 1
                return True

            block_len = hints[h_idx]
            # 計算最晚可能的起始位置
            limit = N - suffix_len[h_idx+1] - block_len
            if h_idx < K - 1: limit -= 1 # 必須留間隔
            
            can_solve = False
            
            # 嘗試所有合法位置
            for p in range(pos, limit + 1):
                # 1. 檢查前置空白 (pos 到 p)
                valid_gap = True
                for i in range(pos, p):
                    if line[i] == FILLED:
                        valid_gap = False; break
                if not valid_gap: break # 遇到 FILLED 不能跳過

                # 2. 檢查區塊 (p 到 p+len)
                valid_block = True
                for i in range(p, p + block_len):
                    if line[i] == EMPTY:
                        valid_block = False; break
                if not valid_block: continue

                # 3. 檢查後置分隔 (p+len)
                next_start = p + block_len
                if h_idx < K - 1:
                    if next_start < N and line[next_start] == FILLED:
                        continue
                    next_start += 1 # 跳過分隔符
                
                # 遞歸
                if dp(h_idx + 1, next_start):
                    can_solve = True
                    # 回溯標記顏色
                    for i in range(pos, p): pos_white[i] = True
                    for i in range(p, p + block_len): pos_black[i] = True
                    if h_idx < K - 1 and p + block_len < N:
                        pos_white[p + block_len] = True
            
            memo[h_idx][pos] = 1 if can_solve else 2
            return can_solve

        if not dp(0, 0): return None

        # 合成新行
        new_line = list(line)
        for i in range(N):
            is_b = pos_black[i]
            is_w = pos_white[i]
            if is_b and not is_w: new_line[i] = FILLED
            elif is_w and not is_b: new_line[i] = EMPTY
            
        return new_line

    def _backtrack(self):
        best_r, best_c = -1, -1
        min_u = 999
        # 啟發式搜索：找未知數最少的行
        for r in range(self.N):
            if UNKNOWN in self.grid[r]:
                cnt = self.grid[r].count(UNKNOWN)
                if cnt < min_u:
                    min_u = cnt
                    for c in range(self.N):
                        if self.grid[r][c] == UNKNOWN:
                            best_r, best_c = r, c; break
        
        if best_r == -1: return True

        saved = deepcopy(self.grid)
        # 嘗試填黑
        self.grid[best_r][best_c] = FILLED
        if self._propagate_queue():
            if self._backtrack(): return True
        
        # 回溯：填白
        self.grid = saved
        self.grid[best_r][best_c] = EMPTY
        if self._propagate_queue():
            if self._backtrack(): return True
        return False

# ======================================================
# 解析器 (嚴格遵守 Column 前, Row 後)
# ======================================================
def parse_input_file_strict(filename):
    puzzles = [] 
    try:
        with open(filename, 'r', encoding='utf-8-sig') as f:
            lines = [line.strip() for line in f]
    except FileNotFoundError:
        print("Input file not found.")
        return []

    idx = 0
    current_pid = None
    buffer = []

    def save_puzzle(pid, buf):
        # 根據規格：
        # 前25筆 -> Column Hints (左到右)
        # 後25筆 -> Row Hints (上到下)
        
        # 先進行基本的數字解析
        parsed_hints = []
        for raw_line in buf:
            # 移除 source 標籤
            clean_line = raw_line.split('[')[0] 
            clean_line = clean_line.split('#')[0]
            # 支援任意空白分隔
            numbers = re.findall(r'\d+', clean_line)
            hints = [int(x) for x in numbers]
            
            # 如果是空行或只包含0，視為 [0] (全白)
            if not hints or (len(hints) == 1 and hints[0] == 0):
                hints = [0]
            elif 0 in hints:
                # 如果有非0又有0，通常是非法格式，但在這裡我們過濾掉0
                hints = [x for x in hints if x != 0]
                if not hints: hints = [0]
                
            parsed_hints.append(hints)

        # 嚴格檢查行數：必須要有 50 行
        if len(parsed_hints) == 50:
            col_hints = parsed_hints[:25]
            row_hints = parsed_hints[25:]
            puzzles.append((pid, col_hints, row_hints))
        else:
            # 如果不是 50 行，嘗試容錯 (例如只有 49 行的 $786)
            # 但在這個嚴格模式下，我們先打印警告
            # print(f"Warning: {pid} has {len(parsed_hints)} lines, skipping strict parsing.")
            
            # 針對 TAAI 可能的特例：如果有 49 行，通常是最後一行空行被省略
            if len(parsed_hints) == 49:
                parsed_hints.append([0])
                puzzles.append((pid, parsed_hints[:25], parsed_hints[25:]))
            # 如果超過 50 行 (例如包含頁碼)，取最後 50 行 (通常頁碼在最前)
            elif len(parsed_hints) > 50:
                final_hints = parsed_hints[-50:]
                puzzles.append((pid, final_hints[:25], final_hints[25:]))

    while idx < len(lines):
        line = lines[idx]
        if '$' in line:
            if current_pid: save_puzzle(current_pid, buffer)
            pid_match = re.search(r'\$(\d+)', line)
            if not pid_match: idx += 1; continue
            current_pid = f"${pid_match.group(1)}"
            buffer = []
            idx += 1
            # 讀取直到下一個 $ 或檔尾
            while idx < len(lines):
                curr_line = lines[idx]
                if '$' in curr_line: break
                idx += 1
                if not curr_line: continue
                buffer.append(curr_line)
        else:
            idx += 1
            
    if current_pid: save_puzzle(current_pid, buffer)
    return puzzles

# ======================================================
# Worker (嚴格模式)
# ======================================================
def solve_single_puzzle(args):
    pid, col_hints, row_hints = args
    
    # 直接求解，不做任何 Swap
    # 因為題目規格說：前面 25 是 Column，後面 25 是 Row
    solver = NonogramSolver(col_hints, row_hints, pid)
    
    # 進行求解
    res = solver.solve()
    
    if res:
        return (pid, res)
    else:
        # 如果嚴格模式解不出來，這裡就是無解 (Unsolvable)
        return (pid, None)

# ======================================================
# 主程式
# ======================================================
def main():
    input_file = 'Nonogram_Q.txt'
    output_file = 'result_final.txt'
    
    print(f"Parsing {input_file} (STRICT MODE: Col first)...")
    all_puzzles = parse_input_file_strict(input_file)
    print(f"Loaded {len(all_puzzles)} puzzles.")
    
    workers = max(1, multiprocessing.cpu_count())
    print(f"Starting STRICT parallel execution with {workers} cores...")
    
    start_t = time.time()
    
    with open(output_file, 'w') as f_out:
        with multiprocessing.Pool(processes=workers) as pool:
            for pid, sol in pool.imap(solve_single_puzzle, all_puzzles):
                if sol:
                    f_out.write(f"{pid}\n")
                    for row in sol:
                        f_out.write("  ".join(["1" if x==1 else "0" for x in row]) + "\n")
                    f_out.flush()
                else:
                    # 失敗時在 console 顯示，但不寫入檔案
                    print(f"Failed: {pid}")
    
    end_t = time.time()
    print(f"All done! Total time: {end_t - start_t:.2f}s")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()