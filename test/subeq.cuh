#pragma once

// Round down(my impl, which is more natural)
// Round up is to float64 and back f8 behavior
int subeq(int a, int b){
    for(int i = 0; i < 4; i++){
        int a0 = (a >> (i * 8)) & 0xff;
        int b0 = (b >> (i * 8)) & 0xff;
        if(std::abs(a0 - b0) > 1){
            return 0;
        }
    }
    return 1;
}