// TODO Implement double/tripple buffering to try and sqeeze out the final ~8% performace

// Explore tensor core ops for the math. Memory ops would still be issue in ampere(no TMA)

// Hypothesis: When we use tensor cores, it might make sence to use a tripple circular buffer with 2 producers to account for slow memory ops
// But this will increase SMEM load and register pressure!! Interesting to see how we can optimize/tune!!