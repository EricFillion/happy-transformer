test_data = [
    {
        'data': ["Who was Jim Henson? Jim [MASK] was a puppeteer.",
                 None,
                 'henson',
                 0.9],
        'id': "without options"
    },
    {
        'data': ["A Toyota Camry is a [MASK], not a truck",
                 ["dog", "john", "white", "car", "van", "truck"],
                 "car",
                 0.9],
        'id': "with options"
    },
]
