let MAXLEN = 300
let WORD_INDEX = JSON.parse($.ajax({
  dataType: "json",
  url: '../../../models/lstm/save/tokenizer_word_index.json',
  async: false
}).responseText);

let tokenize = (text) => {
  text = text.toLowerCase()
  text = text.replace(/[^\w\s'-]/gi, '')
  text = text.replace(/\n/g, '').replace(/\t/g, '')
  return text.split(' ')
}

let text_to_sequence = (text) => {
  let tokens = tokenize(text)
  var indicies = tokens.map(x =>
    WORD_INDEX[x] == 'undefined' ? -1 : WORD_INDEX[x])
  return indicies.filter(x => x != -1)
}

let pad_sequence = (seq, maxlen) => {
  if (seq.length > maxlen)
    return seq.slice(seq.length-maxlen, seq.length)
  for (var i = seq.length; i < maxlen; i++)
    seq.unshift(0)
  return seq
}

let process_text = (text) => {
  return tf.tensor([pad_sequence(text_to_sequence(text), MAXLEN)])
}

// Load trained Keras model
var model;
(async () => {
  model = await tf.loadModel('../../../models/lstm/tfjs/model.json')
})().then(() => {
  $('#prediction #circle').removeClass('loading').addClass('real')
  $('#prediction #helper').text('Ready')
});

async function evaluate(text) {
    var prediction = await model.predict(process_text(text))
    let categories = [{'class': 'toxic', 'label': ['Toxic', 'Non-toxic'], 'score': prediction.get(0, 0)}, 
     {'class': 'severe', 'label': ['Severe', 'Mild'], 'score': prediction.get(0, 1)},
     {'class': 'obscene', 'label': ['Obscene', 'Clean'], 'score': prediction.get(0, 2)},
     {'class': 'threat', 'label': ['Threat', 'Non-threat'], 'score': prediction.get(0, 3)},
     {'class': 'insult', 'label': ['Insult', 'Non-insult'], 'score': prediction.get(0, 4)},
     {'class': 'identity_hate', 'label': ['Identity hate', 'No identity hate'], 'score': prediction.get(0, 5)}]

    var info = {}
    for (var i = 0; i < categories.length; i++) {
	cat = categories[i]
	info[cat.class] = cat.score.toFixed(2)
    }
    console.log(`${text}: ${JSON.stringify(info)}`)

    toxic = categories[0].score
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
