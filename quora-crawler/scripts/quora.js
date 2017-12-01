'use strict';
var QUORA_USER_EMAIL = "<INSERT QUORA USERNAME OR EMAIL HERE>";
var QUORA_PASSWORD = "<INSERT QUORA PASSWORD HERE>";
var fs = require('fs');
var system = require('system');
var content = '',
    f = null,
    lines = null,
	questions = {},
	maxId = 0,
    eol = system.os.name == 'windows' ? "\r\n" : "\n";
try {
    f = fs.open("data/train.csv", "r");
    content = f.read();
} catch (e) {
    console.log(e);
}
if (f) {
    f.close();
}
if (content) {
	console.log("quora.js :: Finished reading the data file.");
	lines = content.split(eol);
    for (var i = 1, len = lines.length; i < len; i++) {
		try {
			if (lines[i].trim()) {
				var lineSplit = lines[i].split('","');
				maxId = parseInt(lineSplit[0].substr(1).slice(0, -1));
				var question1Id = lineSplit[1];
				var question1 = lineSplit[3];
				var question2Id = lineSplit[2];
				var question2 = lineSplit[4];
				questions[question1] = {qUrl : question1.replace(/ /g , "-").slice(0, -1), qId: question1Id};
				questions[question2] = {qUrl : question2.replace(/ /g , "-").slice(0, -1), qId: question2Id};
			}
		} catch (e) {
			console.log(e);
			console.log(lines[i]);
		}
        
    }
	console.log("quora.js :: Total number of questions is " + maxId);
}


var page = require('webpage').create();
page.settings.userAgent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36';
setTimeout(function() {
	page.open('https://www.quora.com/');
	console.log("quora.js :: Opened quora.com");
}, 100);
setTimeout(function(user, pass) {
	page.evaluate(function(user, pass) {
		document.querySelector("input[name='email'][class='text header_login_text_box ignore_interaction']").value = user;
		document.querySelector("input[name='password'][class='text header_login_text_box ignore_interaction']").value = pass;
		document.querySelector("input[value='Login']").click();
	}, user, pass);
	console.log("quora.js :: Logged into quora.com");
}, 5000, QUORA_USER_EMAIL, QUORA_PASSWORD);
setTimeout(function(fs) {
	console.log("quora.js :: Looking for duplicate questions");
	var id = 1;
	for (var question in questions) {
		setTimeout(function(q, id, maxId, qPage, fs) {
			var url = 'https://www.quora.com/' + escape(questions[q].qUrl);
			console.log("quora.js :: Opening URL = " + url);
			qPage.onUrlChanged = function(targetUrl) {
				var dupQuestion = unescape(targetUrl).split("/")[targetUrl.split("/").length - 1].split("?")[0];
				dupQuestion = dupQuestion.replace(/-/g , " ") + "?";
				if (!questions.hasOwnProperty(dupQuestion)) {
					questions[q].qDup = dupQuestion;
					var lineW = '"' + (maxId + id) + '","' + questions[q].qId + '","' + (maxId + id) + '","' + q + '","' + questions[q].qDup + '","1"\n';
					fs.write("data/dup.csv", lineW, 'a');
				}
			};
			qPage.open(url, function(status) {
				if (status !== "success") {
					console.log("quora.js :: Error for " + questions[q].qUrl + " : " + status);
				}
			});
		}, (100 * id), question, id, maxId, page, fs);
		id = id + 1;
	}
}, 10000, fs);