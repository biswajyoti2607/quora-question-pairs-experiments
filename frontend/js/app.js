$(function () {
	
	var dupMap = {};
	var questionMap = {};
	
	$("#check-button").on('click', function() {
		let q1 = $('#firstQuestionInput').dropdown('get value');
		let q2 = $('#secondQuestionInput').dropdown('get value');
		if (dupMap.hasOwnProperty(q1) && dupMap[q1].hasOwnProperty(q2)) {
			$('#resultIcon').removeClass('fa-thumbs-down');
			$('#resultIcon').addClass('fa-thumbs-up');
		} else {
			$('#resultIcon').removeClass('fa-thumbs-up');
			$('#resultIcon').addClass('fa-thumbs-down');
		}
		var $that = $(".icon-wrapper-2");
		$that.css("display","inline-block");
		$that.addClass('anim');
		setTimeout(function() {
			$that.removeClass('anim');
			$that.hide("slow");
		}, 1200);
	});
	
	$.get( "data/data.tsv", function(data) {
		loadData(data);
		
		
		for(let question in questionMap) {
			$('#firstQuestionInput > .menu').append('<div class="item" data-value="' + question + '">' + question + '</div>');
			$('#secondQuestionInput > .menu').append('<div class="item" data-value="' + question + '">' + question + '</div>');
		}
		
		
		$('#firstQuestionInput').dropdown({});
		$('#secondQuestionInput').dropdown({});
		
		$(".whole--container").removeClass("preload");
		setTimeout((function (_this) {
			return function () {
				return $(".whole--container").toggleClass("whole--active", 1000);
			};
		})(this), 100);
	});
	
	function loadData(data) {
		let dataLines = data.split("\n")
		for(let dataItem in dataLines) {
			let itemArr = dataLines[dataItem].split("|*|*|*|");
			if(itemArr.length == 6) {
				let q1 = stripQuotes(itemArr[3]);
				let q2 = stripQuotes(itemArr[4]);
				let isDup = !!+itemArr[5];
				if(!dupMap.hasOwnProperty(q1)) {
					dupMap[q1] = {};
				}
				if(!dupMap.hasOwnProperty(q2)) {
					dupMap[q2] = {};
				}
				dupMap[q1][q1] = true;
				if(isDup == true) {
					dupMap[q1][q2] = true;
					dupMap[q2][q1] = true;
				}
				dupMap[q2][q2] = true;
				questionMap[q1] = q1;
				questionMap[q2] = q2;
			}
		}
	}
	
	function stripQuotes(s) {
		while(s.charAt(0) === '"') {
			s = s.trim().substr(1);
		}
		
		while(s.charAt(s.length - 1) === '"') {
			s = s.trim().substring(0, s.length - 1);
		}
		return s;
	}
});