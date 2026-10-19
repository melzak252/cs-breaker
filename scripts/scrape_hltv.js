/**
 * HLTV Match Results Scraper
 * 
 * Scrapes completed match results from HLTV (https://www.hltv.org/results)
 * using browser automation with offset pagination and polite request pacing.
 * 
 * Output schema matches cs-breaker historical import expectations:
 * match_id, date, team1, team2, tournament, score1, score2, winner, format, match_url
 */

const MONTHS = {
  january: '01', february: '02', march: '03', april: '04', may: '05', june: '06',
  july: '07', august: '08', september: '09', october: '10', november: '11', december: '12'
};

export function parseDateHeadline(headline) {
  if (!headline) return null;
  const m = headline.match(/(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d+)(?:st|nd|rd|th)\s+(\d{4})/i);
  if (!m) return null;
  const month = MONTHS[m[1].toLowerCase()];
  const day = m[2].padStart(2, '0');
  const year = m[3];
  return `${year}-${month}-${day}`;
}

export function extractMatchesFromDocument() {
  const MONTHS_MAP = {
    january: '01', february: '02', march: '03', april: '04', may: '05', june: '06',
    july: '07', august: '08', september: '09', october: '10', november: '11', december: '12'
  };

  function parseDate(h) {
    if (!h) return null;
    const m = h.match(/(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d+)(?:st|nd|rd|th)\s+(\d{4})/i);
    if (!m) return null;
    return `${m[3]}-${MONTHS_MAP[m[1].toLowerCase()]}-${m[2].padStart(2, '0')}`;
  }

  const sublists = Array.from(document.querySelectorAll('.results-sublist'));
  const dates = sublists.map(s => parseDate(s.querySelector('.standard-headline')?.textContent?.trim()));
  const firstValidDate = dates.find(d => d !== null) || new Date().toISOString().slice(0, 10);

  const results = [];
  for (let i = 0; i < sublists.length; i++) {
    const sub = sublists[i];
    const dateStr = dates[i] || firstValidDate;
    const rows = sub.querySelectorAll('.result-con');
    
    for (const r of Array.from(rows)) {
      const a = r.querySelector('a.a-reset');
      const href = a?.getAttribute('href') || '';
      const idMatch = href.match(/\/matches\/(\d+)\//);
      if (!idMatch) continue;
      const matchId = idMatch[1];
      
      const team1Name = r.querySelector('.team1 .team')?.textContent?.trim() || '';
      const team2Name = r.querySelector('.team2 .team')?.textContent?.trim() || '';
      if (!team1Name || !team2Name) continue;
      
      const wonEl = r.querySelector('.score-won');
      const lostEl = r.querySelector('.score-lost');
      const scoreWon = wonEl ? parseInt(wonEl.textContent.trim(), 10) : null;
      const scoreLost = lostEl ? parseInt(lostEl.textContent.trim(), 10) : null;
      if (scoreWon === null || scoreLost === null) continue;
      
      const team1Won = r.querySelector('.team1 .team-won') !== null;
      const team2Won = r.querySelector('.team2 .team-won') !== null;
      
      let score1, score2, winner;
      if (team1Won) {
        score1 = scoreWon;
        score2 = scoreLost;
        winner = 'team1';
      } else if (team2Won) {
        score1 = scoreLost;
        score2 = scoreWon;
        winner = 'team2';
      } else {
        score1 = scoreWon;
        score2 = scoreLost;
        winner = scoreWon > scoreLost ? 'team1' : (scoreLost > scoreWon ? 'team2' : '');
      }
      
      const eventName = r.querySelector('.event-name')?.textContent?.trim() || '';
      const mapText = r.querySelector('.map-text')?.textContent?.trim() || 'bo3';
      
      results.push({
        match_id: matchId,
        date: `${dateStr} 12:00:00`,
        team1: team1Name,
        team2: team2Name,
        tournament: eventName,
        score1: String(score1),
        score2: String(score2),
        winner,
        format: mapText,
        match_url: `https://www.hltv.org${href}`
      });
    }
  }
  return results;
}
