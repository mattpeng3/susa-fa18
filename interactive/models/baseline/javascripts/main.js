`
STOP_WORDS = set(open('./stopwords.txt', 'r').read().split('\n')[:-1])
BAD_WORDS = set(open('./badwords.txt', 'r').read().split('\n')[:-1])

def predict(comment):
    words = len([word for word in comment.split() if word not in STOP_WORDS])
    obscene = len([word for word in comment.split() if word in BAD_WORDS])
    return 0 if words == 0 else obscene/words
`

let tokenize = (text) => {
  text = text.toLowerCase()
  text = text.replace(/[^\w\s'-]/gi, '')
  text = text.replace(/\n/g, '').replace(/\t/g, '')
  return text.split(' ')
}

// Load horrible baseline
var STOP_WORDS
var BAD_WORDS
(async () => {
    $.get('../../../models/baseline/badwords.txt', (data) => {
	BAD_WORDS = new Set(data.split('\n').slice(0, -1))
    }, 'text');
    $.get('../../../models/baseline/stopwords.txt', (data) => {
	STOP_WORDS = new Set(data.split('\n').slice(0, -1))
    }, 'text');
    
})().then(() => {
  $('#prediction #circle').removeClass('loading').addClass('real')
  $('#prediction #helper').text('Ready')
});

function evaluate(text) {
    var words = 0
    var obscene = 0
    var tokens = tokenize(text)
    var toxic = 0
    for (var i = 0; i < tokens.length; i++)
	if (!(STOP_WORDS.has(tokens[i]))) words += 1
    for (var i = 0; i < tokens.length; i++)
	if (BAD_WORDS.has(tokens[i])) obscene += 1
    if (words == 0)
	toxic = 0
    else
	toxic = obscene/words

    if (toxic > 0.5) {
	$('#prediction #circle').removeClass('loading').addClass('fake')
	$('#prediction #helper').text(`Toxic (${toxic.toFixed(2)})`)
    } else {
	$('#prediction #circle').removeClass('loading').addClass('real')
	$('#prediction #helper').text(`Non-toxic (${toxic.toFixed(2)})`)
    }
}

var typingTimer;
$('#news_content').keyup(() => {
  clearTimeout(typingTimer)
  typingTimer = setTimeout(() => {
    $('#prediction #circle').removeClass('real').removeClass('fake').addClass('loading')
    $('#prediction #helper').text('Generating prediction...')
    evaluate($('#news_content').val())
  }, 1000);
})
