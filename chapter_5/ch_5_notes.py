import tensorflow as tf 

# An equivalent way to represent a wave is by examining the 
# frequencies that make it up at each time interval.
# This perspective is called the frequency domain.
# It’s easy to convert between time domain and frequency 
# domains using a mathematical operation called a discrete 
# Fourier transform (commonly implemented using an algorithm
# known as the Fast Fourier transform). We will use this technique
# to extract a feature vector out of our sound.

# A sound may produce 12 kinds of pitches. In music terminology, the
#  12 pitches are C, C#, D, D#, E, F, F#, G, G#, A, A#, and B. 
# Listing 5.2 shows how to retrieve the contribution of each pitch
#  in a 0.1 second interval, resulting in a matrix with 12 rows.
#  The number of columns grows as the length of the audio file increases.
#  Specifically, there will be 10*t columns for a t second audio.
#  This matrix is also called a chromogram of the audio.
