all:
	gcc -oFast bitboard.c -o bbc
	./bbc
	
debug:
	gcc bitboard.c -o bbc
	./bbc
