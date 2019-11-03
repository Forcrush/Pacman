# Pacman (Search Part)

This project is based on Berkeley CS188, the 'search (standard)' can pass all the test in `autograder.py` provided by Berkeley. 'search (high)' is the refined version with less execution time. The entrance of this project is `pacman.py`, more details can be found in [Assignment_1]()

However, the imporved version fails in the `cornersproblem` in `autograder.py`, because the search method to find the goal in bfs/dfs is somewhat different with that in 'search (standard)':

- In 'search (standard)', which is also the standard version to solve problems in Berkeley CS188, the search method to find the goal is continuely searching (if a corner is found, add it to a extra list and put it in stack/queue with current path) until the length of corners list equals to 4.

- In 'search (high)', the search method to find the goal in `cornersproblem` is processing bfs/dfs until meets a corner, then empties all the stack/queue and starts from this corner with bfs/dfs. Iteratively repeats the above steps until all 4 corners are found. 

This issue can be solved to modify the search method in 'search (high)', the modification will be done once I have extra time :).