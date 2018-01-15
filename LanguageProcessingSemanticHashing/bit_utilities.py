import math
import operator

def array_bit_flip(binary_array,index):
    copy_array = binary_array[:]
    copy_array[index] = 1 - copy_array[index]
    return copy_array

def bit_flip(binary_int,index):
    return binary_int ^ 2**index

def convert_encoding_to_bits(binary_array):
    out = 0
    for bit in binary_array:
        out = (out << 1) | bit
    return out

def recursive_compute_hamming_close_codes(code, distance):
    # hamming distance 0
    similar_encodings = [code]

    if distance <= 0:
        return similar_encodings

    recursive_step_compute_hamming_close_codes(code, similar_encodings, 0, distance)


def recursive_step_compute_hamming_close_codes(code, similar_encodings, index, distance):
    if distance <= 0:
        return similar_encodings

    # hamming distance 1
    for i in range(index +1,code.bit_length()):
        flip_one = bit_flip(code, i)
        similar_encodings.append(flip_one)

        if distance > 1:
            recursive_step_compute_hamming_close_codes(flip_one, similar_encodings, i, distance-1)

    return similar_encodings

# complexity is bad
def compute_hamming_close_codes(code,max_distance=5):
    # hamming distance 0
    similar_encodings = [code]


    if max_distance <= 0:
        return similar_encodings

    # hamming distance 1
    for i in range(code.bit_length()):
        flip_one = bit_flip(code, i)
        similar_encodings.append(flip_one)

        if max_distance > 1:
            # hamming distance 2
            for j in range(i+1, flip_one.bit_length()):
                flip_two = bit_flip(flip_one, j)
                similar_encodings.append(flip_two)

                if max_distance > 2:
                    # hamming distance 3
                    for k in range(j+1,flip_two.bit_length()):
                        flip_three = bit_flip(flip_two,k)
                        similar_encodings.append(flip_three)

                        if max_distance > 3:
                            # hamming distance 4
                            for l in range(k + 1, flip_two.bit_length()):
                                flip_four = bit_flip(flip_three, k)
                                similar_encodings.append(flip_four)

                                if max_distance > 4:
                                    # hamming distance 5
                                    for m in range(l + 1, flip_two.bit_length()):
                                        flip_five = bit_flip(flip_four, k)
                                        similar_encodings.append(flip_five)

    return similar_encodings

def bit_count(set_of_bits):
    return bin(set_of_bits).count('1')


def retrieve_docs_within_hamming_distance(code, encoding_to_somethings, max_distance):
    selected_vectors = []
    for encoding in encoding_to_somethings:
        if bit_count(code ^ encoding) <= max_distance:
            selected_vectors += encoding_to_somethings[encoding]

    return selected_vectors



def test_bit_flip():
    bit_str = '10001'
    bit_int =  int(bit_str,2)
    print bin(bit_int)
    print(bin(bit_flip(bit_int, 2)))
    print bin(bit_int)


def test_bit_conversion():
    print(convert_encoding_to_bits([1,0,0,0,1]))

def test_compute_hamming_close_codes():
    bit_str = '10001'
    bit_int = int(bit_str, 2)
    print([bin(b) for b in compute_hamming_close_codes(bit_int,3)])



#test_bit_flip()
#test_compute_hamming_close_codes()