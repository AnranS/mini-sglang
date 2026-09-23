import { setStatus, STATUS_LABEL, STATUS_ORDER, useProgress } from '../../lib/progress';

export default function StageStatus({ id }: { id: string }) {
	const current = useProgress()[id] ?? 'todo';
	return (
		<div className="stage-status" role="radiogroup" aria-label="这一阶段的学习进度">
			{STATUS_ORDER.map((status) => (
				<button
					key={status}
					type="button"
					role="radio"
					aria-checked={current === status}
					data-status={status}
					className="stage-status-option"
					onClick={() => setStatus(id, status)}
				>
					{STATUS_LABEL[status]}
				</button>
			))}
		</div>
	);
}
