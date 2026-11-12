"""Automated checks for conservative corpus triage; no approval claims."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from amn_nmc.data_audit import audit,inspect,digest

def test_bad_wiki_sections_and_url_are_flagged():
    text='Türkiye ve Türkçe dil öğrenimi üzerine açıklamalar. '*15+'\nKaynakça\nhttps://example.org'
    _,flags=inspect(text)
    assert 'reference_sections' in flags
    assert 'url_in_body' in flags

def test_foreign_english_text_cannot_pass_unnoticed():
    text='This is entirely English with syntax examples and foreign words. '*18
    _,flags=inspect(text)
    assert 'weak_turkish' in flags

def test_raw_jsonl_duplicate_privacy_and_qa_rejection(tmp_path):
    raw=tmp_path/'source.jsonl'
    text='Türkçe dil, yazılı ve sözlü biçimlerde anlam üretir. '+('Bu durum farklı kaynaklarda ayrıntılı anlatılır. '*15)
    docs=[
       {'text':text,'source':'https://tr.wikipedia.org/wiki/Deneme'},
       {'text':text,'source':'https://tr.wikipedia.org/wiki/Deneme_2'},
       {'text':text+' İletişim: kisi@example.com','source':'https://tr.wikipedia.org/wiki/Deneme_3'},
       {'prompt':'Ne?','answer':'Hazır cevap'}
    ]
    raw.write_text('\n'.join(json.dumps(row,ensure_ascii=False) for row in docs),encoding='utf-8')
    result=audit(raw,tmp_path/'quality',declared_license='CC-BY-SA-4.0')
    assert result['total_rows']==4
    assert result['flag_counts']['exact_duplicate']>=1
    assert result['flag_counts']['email']==1
    assert result['flag_counts']['missing_raw_text']==1
    assert result['candidate_is_approved_for_training'] is False
    assert (tmp_path/'quality'/'candidate_only.jsonl').exists()

def test_invalid_utf8_is_recorded(tmp_path):
    raw=tmp_path/'bad.jsonl'
    raw.write_bytes(b'{"text":"\xff"}\n')
    result=audit(raw,tmp_path/'quality')
    assert result['flag_counts']['invalid_utf8_or_json']==1

def test_canonical_hash_retains_turkish_distinctions():
    assert digest('TÜRKÇE\n DENEME')==digest('türkçe deneme')
    assert digest('sık') != digest('sik')

def test_wiki_sample_untruncated_and_unapproved(tmp_path):
    from amn_nmc.wiki_sample_audit import sample_and_audit
    full=('Türkçe dil ve anlambilim konusunda bu metin açıklayıcı bilgi içerir. '*30)
    def fetch(url):
        return {'rows':[{'row':{'text':full+str(i),
            'source':f'https://tr.wikipedia.org/wiki/Madde_{i}',
            'page_id':i,'revision_id':42,'title':f'Madde {i}'}} for i in range(12)]}
    info=sample_and_audit(tmp_path/'remote',offsets=(0,),length=12,fetch=fetch)
    assert info['sampled_articles']==12
    assert info['approved_for_training'] is False
    source=tmp_path/'remote'/'sampled_untruncated.jsonl'
    assert json.loads(source.read_text().splitlines()[0])['text']==full+'0'
    assert (tmp_path/'remote'/'quality'/'audit.json').exists()
