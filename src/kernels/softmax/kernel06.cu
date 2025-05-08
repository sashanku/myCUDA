// TODO Explore block tiling to improve arithmetic intensity


// Instead of having one block calculate only one row of the output matrix, one block
// will compute `TILE_SIZE` number of rows. This way we will have fewer blocks, and more
// computations per block. 2D threads will process elements. `tx` will process elements and
// `ty` will process rows in a block

// We will need partial block-wide reduction with width `tx` threads participating in the reduction
// for each row's maximum and norm value.
