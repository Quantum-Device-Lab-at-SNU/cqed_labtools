import cqed_labtools

measured_data_set = {
    'q0': {
        'f_resonator_ground': 6.0820e9,
        'punchout_shift': 33.042e6,
        'f_qubit_dressed': 6.955e9,
        'anharmonicity_dressed': -129e6
    },
    'q1': {
        'f_resonator_ground': 6.2666e9,
        'punchout_shift': 21.451e6,
        'f_qubit_dressed': 7.570e9,
        'anharmonicity_dressed': -132e6
    },
    'q2': {
        'f_resonator_ground': 6.3894e9,
        'punchout_shift': 16.30e6,
        'f_qubit_dressed': 8.106e9,
        'anharmonicity_dressed': -134e6
    },
    'q3': {
        'f_resonator_ground': 6.5678e9,
        'punchout_shift': 8.475e6,
        'f_qubit_dressed': 8.814e9,
        'anharmonicity_dressed': -128e6
    },
    'q4': {
        'f_resonator_ground': 6.997e9,
        'punchout_shift': 9.053e6,
        'f_qubit_dressed': 9.549e9,
        'anharmonicity_dressed': -124e6
    },
    'q5': {
        'f_resonator_ground': 7.099e9,
        'punchout_shift': 6.613e6,
        'f_qubit_dressed': 10.114e9,
        'anharmonicity_dressed': -120e6
    }
}

extracted_data = {}
for _q in measured_data_set.keys():
    q_data = measured_data_set[_q]
    tc = cqed_labtools.TransmonCircuit.from_dressed_freqs_with_punchout_shift(
        q_data['f_qubit_dressed'],
        q_data['f_resonator_ground'],
        q_data['anharmonicity_dressed'],
        q_data['punchout_shift'],
        solver='exact'
    )

    extracted_data[_q] = {
        'EJ_over_h': tc.transmon.EJ_over_h,
        'EC_over_h': tc.transmon.EC_over_h,
        'g_over_2pi': tc.g_over_2pi,
        'EJ_over_EC': tc.transmon.EJ_over_EC,
        'f_qubit_bare': tc.f_qubit(0, representation='bare'),
        'f_resonator_bare': tc.f_resonator(representation='bare')
    }
